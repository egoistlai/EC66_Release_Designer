"""EMCP phase-1 planners for a complete p/R/v/omega terminal state.

The tool point, full world-frame twist convention, local-terminal B-spline and
native Ruckig conversion intentionally match the portable EMCP implementation.
Newton remains the only particle-retention acceptance test.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.interpolate import BSpline, PPoly
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

from kinematics import EC66Kinematics, mount_transform, transform
from trajectory_planning import BucketCollisionModel, Plan, json_safe, quintic_motion


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ppoly_from_bspline(spline: BSpline, start: float, stop: float) -> PPoly:
    curves = [PPoly.from_spline((spline.t, spline.c[:, joint], spline.k)) for joint in range(6)]
    knots = np.unique(np.r_[start, curves[0].x[(curves[0].x > start) & (curves[0].x < stop)], stop])
    coefficients = np.zeros((spline.k + 1, len(knots) - 1, 6), dtype=float)
    for interval, left in enumerate(knots[:-1]):
        for derivative in range(spline.k + 1):
            coefficients[spline.k - derivative, interval] = np.asarray(
                [curve(left, derivative) for curve in curves]
            ) / math.factorial(derivative)
    return PPoly(coefficients, knots)


def local_terminal_motion(
    q0: np.ndarray,
    q1: np.ndarray,
    qd1: np.ndarray,
    duration: float,
    terminal_fraction: float,
) -> PPoly:
    """EMCP M23: one global C4 quintic with knots concentrated near the end."""
    join = 1.0 - float(terminal_fraction)
    inside = np.r_[
        np.linspace(0.0, join, 17)[1:-1],
        join,
        join + terminal_fraction / 3.0,
        join + 2.0 * terminal_fraction / 3.0,
    ]
    base = BSpline(
        np.r_[np.zeros(6), np.ones(6)],
        np.stack([q0, q0, q0, q1, q1, q1]),
        5,
    )
    for knot in inside:
        base = base.insert_knot(float(knot))
    controls = base.c.copy()
    basis = BSpline(base.t, np.eye(len(controls)), 5)
    controls[-2] = q1 - qd1 * float(duration) / basis(1.0, nu=1)[-1]
    second = basis(1.0, nu=2)
    controls[-3] -= (second @ controls) / second[-3]
    normalized = BSpline(base.t, controls, 5)
    return _ppoly_from_bspline(
        BSpline(normalized.t * float(duration), normalized.c, normalized.k),
        0.0,
        float(duration),
    )


def ruckig_motion(
    q0: np.ndarray,
    q1: np.ndarray,
    qd1: np.ndarray,
    velocity_limits: np.ndarray,
    acceleration_limits: np.ndarray,
    jerk_limits: np.ndarray,
    minimum_duration: float | None,
    *,
    current_velocity: np.ndarray | None = None,
    current_acceleration: np.ndarray | None = None,
) -> tuple[PPoly, dict]:
    """Calculate native Ruckig motion and preserve every constant-jerk piece."""
    import ruckig

    zero = np.zeros(6)
    inp = ruckig.InputParameter(6)
    inp.current_position = q0.tolist()
    inp.current_velocity = np.asarray(
        zero if current_velocity is None else current_velocity, dtype=float
    ).tolist()
    inp.current_acceleration = np.asarray(
        zero if current_acceleration is None else current_acceleration, dtype=float
    ).tolist()
    inp.target_position = q1.tolist()
    inp.target_velocity = qd1.tolist()
    inp.target_acceleration = zero.tolist()
    inp.max_velocity = velocity_limits.tolist()
    inp.max_acceleration = acceleration_limits.tolist()
    inp.max_jerk = jerk_limits.tolist()
    if minimum_duration is not None and minimum_duration > 0.0:
        inp.minimum_duration = float(minimum_duration)
    trajectory = ruckig.Trajectory(6)
    result = ruckig.Ruckig(6).calculate(inp, trajectory)
    if result not in (ruckig.Result.Working, ruckig.Result.Finished):
        raise RuntimeError(f"Ruckig failed with {result}")

    knots = [0.0, float(trajectory.duration)]
    for profile in trajectory.profiles[0]:
        knots.extend(np.cumsum(profile.t).tolist())
    knots = np.sort(np.asarray(knots, dtype=float))
    knots = knots[np.r_[True, np.diff(knots) > 1.0e-9]]
    knots = knots[(knots >= 0.0) & (knots <= trajectory.duration + 1.0e-9)]
    knots[-1] = trajectory.duration
    coefficients = np.zeros((4, len(knots) - 1, 6), dtype=float)
    for index, (left, right) in enumerate(zip(knots[:-1], knots[1:])):
        position, velocity, acceleration = map(np.asarray, trajectory.at_time(float(left)))
        sample = 0.25 * (right - left)
        jerk = (
            np.asarray(trajectory.at_time(float(left + 2.0 * sample))[2])
            - np.asarray(trajectory.at_time(float(left + sample))[2])
        ) / sample
        coefficients[:, index] = np.stack((jerk / 6.0, acceleration / 2.0, velocity, position))
    motion = PPoly(coefficients, knots)
    probe = np.linspace(0.0, trajectory.duration, 101)
    for derivative in range(3):
        reference = np.asarray([trajectory.at_time(float(t))[derivative] for t in probe])
        np.testing.assert_allclose(motion(probe, derivative), reference, atol=1.0e-7)
    return motion, {
        "ruckig_version": getattr(ruckig, "__version__", "0.19.4"),
        "native_duration_s": float(trajectory.duration),
        "constant_jerk_piece_count": int(len(knots) - 1),
    }


@dataclass(frozen=True)
class EndpointWitness:
    q: np.ndarray
    qd: np.ndarray
    pose_error_m: float
    orientation_error_deg: float
    linear_velocity_error_m_s: float
    angular_velocity_error_rad_s: float


class EMCPPhase1Context:
    def __init__(
        self,
        config_path: Path | None = None,
        *,
        roll_about_normal_deg: float = 0.0,
        q_start_rad: np.ndarray | None = None,
        target: dict | None = None,
        tolerances: dict | None = None,
        endpoint_trial_count: int = 25,
        endpoint_max_solutions: int | None = None,
        endpoint_seed_q_rad: np.ndarray | None = None,
    ):
        self.config_path = (config_path or (HERE / "emcp_phase1_config.json")).resolve()
        self.settings = copy.deepcopy(_load(self.config_path))
        if tolerances:
            self.settings.update({key: float(value) for key, value in tolerances.items()})
        self.scene = _load(HERE / "scene_config.json")
        self.q_start = (
            np.asarray(q_start_rad, dtype=float).reshape(6).copy()
            if q_start_rad is not None
            else np.deg2rad(np.asarray(self.settings["q_start_deg"], dtype=float))
        )
        self.tool_reference = np.asarray(self.settings["tool_reference_local_m"], dtype=float)
        target_values = target or self.settings["target"]
        self.target_position = np.asarray(target_values["p_world_m"], dtype=float)
        self.base_target_rotation = np.asarray(target_values["R_world_from_tool"], dtype=float)
        self.roll_about_normal_deg = float(roll_about_normal_deg)
        self.target_rotation = self.base_target_rotation @ Rotation.from_euler(
            "z", np.deg2rad(self.roll_about_normal_deg)
        ).as_matrix()
        self.target_linear_velocity = np.asarray(target_values["v_world_m_s"], dtype=float)
        self.target_angular_velocity = np.asarray(target_values["omega_world_rad_s"], dtype=float)
        np.testing.assert_allclose(self.target_rotation.T @ self.target_rotation, np.eye(3), atol=1.0e-8)
        self.joint_velocity_limits = np.deg2rad(
            np.asarray(self.settings["joint_velocity_limits_deg_s"], dtype=float)
        )
        self.joint_acceleration_limits = np.deg2rad(
            np.asarray(self.settings["joint_acceleration_limits_deg_s2"], dtype=float)
        )
        self.joint_jerk_limits = np.asarray(self.settings["joint_jerk_limits_rad_s3"], dtype=float)
        self.kinematics = EC66Kinematics(
            HERE / self.scene["urdf"],
            transform(self.scene["world_from_base_xyz"], np.deg2rad(self.scene["world_from_base_rpy_deg"])),
            mount_transform(self.scene["mount_rotations_deg"], self.scene["mount_xyz"]),
            self.scene["shovel_normal_local"],
        )
        self.start_state_metadata = self._validate_configured_start_state(
            validate_flange_pose=q_start_rad is None
        )
        collision_settings = dict(self.settings)
        self.collision = BucketCollisionModel(self.kinematics, self.scene, collision_settings)
        self.rng = np.random.default_rng(int(self.settings["random_seed"]))
        self.endpoint_seed_q_rad = (
            None if endpoint_seed_q_rad is None
            else np.clip(
                np.asarray(endpoint_seed_q_rad, dtype=float).reshape(6),
                self.kinematics.low + 1.0e-6,
                self.kinematics.high - 1.0e-6,
            )
        )
        self.endpoint_trials: list[dict] = []
        self.endpoint_candidates = self._solve_endpoint_candidates(
            int(endpoint_trial_count), endpoint_max_solutions
        )
        self.endpoint = self.endpoint_candidates[0]
        self.q_goal = self.endpoint.q
        self.qd_goal = self.endpoint.qd

    def _validate_configured_start_state(self, *, validate_flange_pose: bool) -> dict:
        """Record and, for configured starts, verify base-frame flange semantics."""
        flange_base = self.kinematics.flange_pose_base(self.q_start)
        flange_quaternion = Rotation.from_matrix(flange_base[:3, :3]).as_quat()
        tool_world = self.kinematics.tool_pose_world(
            self.q_start, self.tool_reference
        )
        metadata = {
            "semantics": "robot_base_frame_flange_tcp_pose",
            "q_start_deg": np.rad2deg(self.q_start),
            "flange_tcp_pose_base": {
                "p_base_m": flange_base[:3, 3],
                "quaternion_xyzw": flange_quaternion,
            },
            "configured_shovel_tcp_pose_world": {
                "p_world_m": tool_world[:3, 3],
                "quaternion_xyzw": Rotation.from_matrix(
                    tool_world[:3, :3]
                ).as_quat(),
                "tool_reference_local_m": self.tool_reference,
            },
        }
        expected = self.scene.get("initial_flange_tcp_pose_base")
        if validate_flange_pose and expected is not None:
            expected_position = np.asarray(expected["p_base_m"], dtype=float)
            expected_rotation = Rotation.from_quat(
                expected["quaternion_xyzw"]
            ).as_matrix()
            position_error = float(
                np.linalg.norm(flange_base[:3, 3] - expected_position)
            )
            orientation_error = float(
                np.degrees(
                    Rotation.from_matrix(
                        flange_base[:3, :3] @ expected_rotation.T
                    ).magnitude()
                )
            )
            metadata["configured_pose_error_m"] = position_error
            metadata["configured_orientation_error_deg"] = orientation_error
            if position_error > 1.0e-5 or orientation_error > 1.0e-2:
                raise ValueError(
                    "q_start_deg does not match scene_config "
                    "initial_flange_tcp_pose_base: "
                    f"position error={position_error:.6g} m, "
                    f"orientation error={orientation_error:.6g} deg"
                )
        return metadata

    def _solve_endpoint_candidates(
        self, trial_count: int = 25, max_solutions: int | None = None
    ) -> list[EndpointWitness]:
        position_tol = float(self.settings["goal_position_tolerance_m"])
        orientation_tol = float(self.settings["goal_orientation_tolerance_deg"])
        candidates: list[EndpointWitness] = []
        scales = (0.25, 0.65, 1.1, 1.7)
        for index in range(max(1, int(trial_count))):
            if self.endpoint_seed_q_rad is not None and index == 0:
                seed = self.endpoint_seed_q_rad.copy()
            elif index == (1 if self.endpoint_seed_q_rad is not None else 0):
                seed = self.q_start.copy()
            else:
                random_index = index - (2 if self.endpoint_seed_q_rad is not None else 1)
                seed = np.clip(
                    self.q_start + self.rng.normal(0.0, scales[random_index % len(scales)], 6),
                    self.kinematics.low + 1.0e-6,
                    self.kinematics.high - 1.0e-6,
                )
            result = self.kinematics.solve_full_pose(
                seed,
                self.target_position,
                self.target_rotation,
                tool_reference_local_m=self.tool_reference,
                max_evaluations=260,
                position_tolerance_m=position_tol,
                orientation_tolerance_deg=orientation_tol,
            )
            q = result.q_rad.copy()
            for joint in range(6):
                choices = [
                    q[joint] + 2.0 * np.pi * winding
                    for winding in range(-2, 3)
                    if self.kinematics.low[joint] + 1.0e-6
                    <= q[joint] + 2.0 * np.pi * winding
                    <= self.kinematics.high[joint] - 1.0e-6
                ]
                if choices:
                    q[joint] = min(choices, key=lambda value: abs(value - self.q_start[joint]))
            jacobian = self.kinematics.spatial_jacobian(q, self.tool_reference)
            desired_twist = np.r_[self.target_linear_velocity, self.target_angular_velocity]
            velocity_bound = (
                float(self.settings["joint_velocity_hard_fraction"]) * self.joint_velocity_limits
            )
            velocity = lsq_linear(
                jacobian,
                desired_twist,
                bounds=(-velocity_bound, velocity_bound),
                tol=1.0e-12,
                lsmr_tol=1.0e-12,
            ).x
            achieved = jacobian @ velocity
            self_clearance, self_pair = self.collision.self_clearance(q)
            record = {
                "trial": index,
                "q_rad": q,
                "pose_reachable": result.reachable,
                "optimizer_success": result.optimizer_success,
                "position_error_m": result.position_error_m,
                "orientation_error_deg": result.orientation_error_deg,
                "linear_velocity_error_m_s": float(np.linalg.norm(achieved[:3] - desired_twist[:3])),
                "angular_velocity_error_rad_s": float(np.linalg.norm(achieved[3:] - desired_twist[3:])),
                "minimum_bucket_clearance_m": self.collision.min_clearance(q),
                "minimum_self_clearance_m": self_clearance,
                "minimum_self_collision_pair": self_pair,
            }
            self.endpoint_trials.append(json_safe(record))
            witness = EndpointWitness(
                q=q,
                qd=np.asarray(velocity),
                pose_error_m=float(result.position_error_m),
                orientation_error_deg=float(result.orientation_error_deg),
                linear_velocity_error_m_s=record["linear_velocity_error_m_s"],
                angular_velocity_error_rad_s=record["angular_velocity_error_rad_s"],
            )
            if (
                result.reachable
                and record["linear_velocity_error_m_s"]
                <= float(self.settings["goal_linear_velocity_tolerance_m_s"])
                and record["angular_velocity_error_rad_s"]
                <= float(self.settings["goal_angular_velocity_tolerance_rad_s"])
                and record["minimum_bucket_clearance_m"] >= float(self.settings["collision_clearance_m"])
                and record["minimum_self_clearance_m"]
                >= float(self.settings.get("self_collision_clearance_m", 0.0))
                and not any(np.linalg.norm(q - old.q) < 1.0e-4 for old in candidates)
            ):
                candidates.append(witness)
        if not candidates:
            raise RuntimeError("No full p/R/v/omega endpoint witness passed the finite multi-start search")
        candidates.sort(
            key=lambda item: float(
                np.linalg.norm(item.q - self.q_start)
                + 0.05 * np.linalg.norm(item.qd / self.joint_velocity_limits)
            )
        )
        return candidates if max_solutions is None else candidates[: max(1, int(max_solutions))]

    def _solve_endpoint(self) -> EndpointWitness:
        """Backward-compatible single-witness endpoint solver."""
        return self._solve_endpoint_candidates()[0]

    def audit_motion(self, motion: PPoly) -> dict:
        duration = float(motion.x[-1] - motion.x[0])
        step = float(self.settings["trajectory_sample_step_s"])
        times = np.linspace(0.0, duration, max(81, int(np.ceil(duration / step)) + 1))
        q = np.asarray(motion(times))
        qd = np.asarray(motion(times, 1))
        qdd = np.asarray(motion(times, 2))
        poses = np.asarray([self.kinematics.tool_pose_world(value, self.tool_reference) for value in q])
        positions = poses[:, :3, 3]
        rotations = poses[:, :3, :3]
        normals = rotations @ self.kinematics.shovel_normal_local
        twists = np.asarray(
            [self.kinematics.tool_twist_world(a, b, self.tool_reference) for a, b in zip(q, qd)]
        )
        dt = times[1] - times[0]
        accelerations = np.gradient(twists, dt, axis=0, edge_order=2)
        clearances = np.asarray([self.collision.min_clearance(value) for value in q])
        self_step = float(self.settings.get("self_collision_sample_step_s", step))
        self_stride = max(1, int(np.floor(self_step / max(dt, 1.0e-12))))
        self_indices = np.unique(np.r_[np.arange(0, len(q), self_stride), len(q) - 1]).astype(int)
        self_clearance_records = [self.collision.self_clearance(q[index]) for index in self_indices]
        self_clearances = np.asarray([record[0] for record in self_clearance_records])
        minimum_self_index = int(np.argmin(self_clearances))
        path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
        displacement = float(np.linalg.norm(positions[-1] - positions[0]))
        joint_travel = np.rad2deg(np.sum(np.abs(np.diff(q, axis=0)), axis=0))
        velocity_ratio = float(np.max(np.abs(qd) / self.joint_velocity_limits))
        acceleration_ratio = float(np.max(np.abs(qdd) / self.joint_acceleration_limits))
        tcp_ratios = {
            "linear_velocity": float(np.linalg.norm(twists[:, :3], axis=1).max() / self.settings["tcp_linear_velocity_limit_m_s"]),
            "linear_acceleration": float(np.linalg.norm(accelerations[:, :3], axis=1).max() / self.settings["tcp_linear_acceleration_limit_m_s2"]),
            "angular_velocity": float(np.linalg.norm(twists[:, 3:], axis=1).max() / self.settings["tcp_angular_velocity_limit_rad_s"]),
            "angular_acceleration": float(np.linalg.norm(accelerations[:, 3:], axis=1).max() / self.settings["tcp_angular_acceleration_limit_rad_s2"]),
        }
        position_error = float(np.linalg.norm(positions[-1] - self.target_position))
        orientation_error = float(
            np.degrees(Rotation.from_matrix(rotations[-1] @ self.target_rotation.T).magnitude())
        )
        linear_error = float(np.linalg.norm(twists[-1, :3] - self.target_linear_velocity))
        angular_error = float(np.linalg.norm(twists[-1, 3:] - self.target_angular_velocity))
        direction_error = float(
            np.degrees(
                np.arccos(
                    np.clip(
                        np.dot(twists[-1, :3], self.target_linear_velocity)
                        / max(
                            np.linalg.norm(twists[-1, :3])
                            * np.linalg.norm(self.target_linear_velocity),
                            1.0e-12,
                        ),
                        -1.0,
                        1.0,
                    )
                )
            )
        )
        apparent_force = np.array([0.0, 0.0, -9.81]) - accelerations[:, :3]
        support = -np.sum(apparent_force * normals, axis=1)
        tangent = np.linalg.norm(apparent_force + support[:, None] * normals, axis=1)
        force_violation = float(
            np.mean(np.maximum(tangent - 0.4 * support + 0.3, 0.0))
            + np.mean(np.maximum(0.5 - support, 0.0))
        )
        terminal_mask = times >= 0.8 * duration
        workspace_min = np.asarray(self.settings["workspace_min_world_m"])
        workspace_max = np.asarray(self.settings["workspace_max_world_m"])
        spatial_valid = bool(
            np.all(positions >= workspace_min)
            and np.all(positions <= workspace_max)
            and path_length <= float(self.settings["max_tcp_path_length_m"])
            and path_length / max(displacement, 0.02) <= float(self.settings["max_tcp_detour_ratio"])
            and np.all(joint_travel <= np.asarray(self.settings["max_joint_total_travel_deg"]))
        )
        valid = bool(
            np.all(q >= self.kinematics.low - 1.0e-9)
            and np.all(q <= self.kinematics.high + 1.0e-9)
            and velocity_ratio <= float(self.settings["joint_velocity_hard_fraction"]) + 1.0e-3
            and acceleration_ratio <= float(self.settings["joint_acceleration_hard_fraction"]) + 1.0e-3
            and max(tcp_ratios.values()) <= 1.0 + 5.0e-3
            and float(clearances.min()) >= float(self.settings["collision_clearance_m"])
            and float(self_clearances.min()) >= float(
                self.settings.get("self_collision_clearance_m", 0.0)
            )
            and position_error <= float(self.settings["goal_position_tolerance_m"])
            and orientation_error <= float(self.settings["goal_orientation_tolerance_deg"])
            and linear_error <= float(self.settings["goal_linear_velocity_tolerance_m_s"])
            and angular_error <= float(self.settings["goal_angular_velocity_tolerance_rad_s"])
            and spatial_valid
        )
        return json_safe(
            {
                "valid": valid,
                "duration_s": duration,
                "sample_count": len(times),
                "joint_velocity_ratio": velocity_ratio,
                "joint_acceleration_ratio": acceleration_ratio,
                "tcp_ratios": tcp_ratios,
                "tcp_max_ratio": max(tcp_ratios.values()),
                "minimum_bucket_clearance_m": float(clearances.min()),
                "minimum_self_clearance_m": float(self_clearances.min()),
                "minimum_self_collision_pair": self_clearance_records[minimum_self_index][1],
                "minimum_self_clearance_time_s": float(times[self_indices[minimum_self_index]]),
                "self_collision_sample_count": len(self_indices),
                "bucket_collision_free": bool(clearances.min() >= self.settings["collision_clearance_m"]),
                "self_collision_free": bool(
                    self_clearances.min() >= self.settings.get("self_collision_clearance_m", 0.0)
                ),
                "collision_free": bool(
                    clearances.min() >= self.settings["collision_clearance_m"]
                    and self_clearances.min() >= self.settings.get("self_collision_clearance_m", 0.0)
                ),
                "collision_scope": "sampled EC66/shovel versus pot plus non-adjacent URDF-link and shovel-to-arm OBB self-collision",
                "path_length_m": path_length,
                "path_detour_ratio": path_length / max(displacement, 0.02),
                "joint_total_travel_deg": joint_travel,
                "spatial_valid": spatial_valid,
                "goal_position_error_m": position_error,
                "goal_orientation_error_deg": orientation_error,
                "goal_linear_velocity_error_m_s": linear_error,
                "goal_angular_velocity_error_rad_s": angular_error,
                "terminal_velocity_world_m_s": twists[-1, :3],
                "terminal_angular_velocity_world_rad_s": twists[-1, 3:],
                "terminal_speed_m_s": float(np.linalg.norm(twists[-1, :3])),
                "terminal_velocity_direction_error_deg": direction_error,
                "terminal_angular_speed_rad_s": float(np.linalg.norm(twists[-1, 3:])),
                "terminal_window_mean_tangential_apparent_acceleration_m_s2": float(np.mean(tangent[terminal_mask])),
                "terminal_window_mean_angular_speed_rad_s": float(np.mean(np.linalg.norm(twists[terminal_mask, 3:], axis=1))),
                "emcp_nominal_centroid_force_violation": force_violation,
                "emcp_inspired_prefilter_score": force_violation + 0.15 * path_length + 0.03 * duration,
                "start_velocity_max_deg_s": float(np.max(np.abs(np.rad2deg(qd[0])))),
                "start_acceleration_max_deg_s2": float(np.max(np.abs(np.rad2deg(qdd[0])))),
                "terminal_acceleration_max_deg_s2": float(np.max(np.abs(np.rad2deg(qdd[-1])))),
            }
        )

    def _plan(self, planner_id: str, label: str, family: str, motion: PPoly, detail: dict, started: float) -> Plan:
        audit = self.audit_motion(motion)
        audit["candidate_parameters"] = json_safe(detail)
        return Plan(
            planner_id=planner_id,
            label=label,
            family=family,
            motion=motion,
            waypoints=np.vstack((self.q_start, self.q_goal)),
            planning_seconds=time.perf_counter() - started,
            audit=audit,
        )

    def generate_candidates(self) -> tuple[list[Plan], list[dict]]:
        all_plans: list[Plan] = []
        failures: list[dict] = []
        for duration in self.settings["quintic_durations_s"]:
            started = time.perf_counter()
            motion = quintic_motion(self.q_start, self.q_goal, self.qd_goal, float(duration))
            all_plans.append(self._plan(f"quintic_D{duration:g}", f"Global quintic D={duration:g}s", "quintic", motion, {"duration_s": duration}, started))
        for fraction in self.settings["local_terminal_fractions"]:
            for duration in self.settings["local_terminal_durations_s"]:
                started = time.perf_counter()
                motion = local_terminal_motion(self.q_start, self.q_goal, self.qd_goal, float(duration), float(fraction))
                all_plans.append(self._plan(f"local_f{int(round(100*fraction)):02d}_D{duration:g}", f"Local terminal {fraction:.0%}, D={duration:g}s", "local_terminal", motion, {"terminal_fraction": fraction, "duration_s": duration}, started))
        for jerk in self.settings["ruckig_jerk_profiles_rad_s3"]:
            for minimum in self.settings["ruckig_minimum_durations_s"]:
                started = time.perf_counter()
                try:
                    motion, detail = ruckig_motion(
                        self.q_start,
                        self.q_goal,
                        self.qd_goal,
                        float(self.settings["joint_velocity_hard_fraction"]) * self.joint_velocity_limits,
                        float(self.settings["joint_acceleration_hard_fraction"]) * self.joint_acceleration_limits,
                        np.full(6, float(jerk)),
                        None if float(minimum) <= 0.0 else float(minimum),
                    )
                    detail.update(jerk_limit_profile_rad_s3=jerk, minimum_duration_s=minimum)
                    all_plans.append(self._plan(f"ruckig_J{jerk:g}_D{minimum:g}", f"Ruckig J={jerk:g}, Dmin={minimum:g}s", "ruckig", motion, detail, started))
                except Exception as error:
                    failures.append({"family": "ruckig", "jerk": jerk, "minimum_duration": minimum, "error": repr(error)})

        selected: list[Plan] = []
        per_method = int(self.settings["max_candidates_per_method"])
        for family in ("quintic", "local_terminal", "ruckig"):
            valid = [plan for plan in all_plans if plan.family == family and plan.audit["valid"]]
            valid.sort(key=lambda plan: plan.audit["emcp_inspired_prefilter_score"])
            selected.extend(valid[:per_method])
        failures.extend(
            {
                "id": plan.planner_id,
                "family": plan.family,
                "status": "mechanical_or_terminal_rejected",
                "audit": plan.audit,
            }
            for plan in all_plans
            if not plan.audit["valid"]
        )
        if not selected:
            raise RuntimeError("No phase-1 candidate passed the full terminal/mechanical audit")
        return selected, failures

    def save_plan(self, plan: Plan, output_dir: Path) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{plan.planner_id}.npz"
        duration = float(plan.motion.x[-1])
        sample_times = np.linspace(0.0, duration, max(2, int(np.ceil(duration * 60.0)) + 1))
        q = np.asarray(plan.motion(sample_times))
        poses = []
        tool_positions = []
        for angles in q:
            flange = self.kinematics.flange_pose_base(angles)
            poses.append(np.r_[flange[:3, 3], Rotation.from_matrix(flange[:3, :3]).as_quat()])
            tool_positions.append(
                self.kinematics.tool_pose_world(angles, self.tool_reference)[:3, 3]
            )
        metadata = {
            "planner_id": plan.planner_id,
            "family": plan.family,
            "full_terminal_target": {
                "p_world_m": self.target_position,
                "R_world_from_tool": self.target_rotation,
                "v_world_m_s": self.target_linear_velocity,
                "omega_world_rad_s": self.target_angular_velocity,
                "roll_about_normal_deg": self.roll_about_normal_deg,
            },
            "tool_reference_local_m": self.tool_reference,
            "audit": plan.audit,
        }
        np.savez_compressed(
            path,
            coefficients=plan.motion.c,
            t=plan.motion.x,
            q=q,
            poses=np.asarray(poses),
            tool_positions_world_m=np.asarray(tool_positions),
            errors=np.zeros((len(q), 2)),
            release_start_s=duration,
            release_action_start_s=duration,
            release_state_s=duration,
            terminal_qd_rad_s=np.asarray(plan.motion(duration, 1)),
            tool_reference_local_m=self.tool_reference,
            target_p_world_m=self.target_position,
            target_R_world_from_tool=self.target_rotation,
            target_v_world_m_s=self.target_linear_velocity,
            target_omega_world_rad_s=self.target_angular_velocity,
            metadata_json=json.dumps(json_safe(metadata), ensure_ascii=False),
        )
        return {
            "id": plan.planner_id,
            "label": plan.label,
            "family": plan.family,
            "path": str(path.resolve()),
            "planning_seconds": plan.planning_seconds,
            "audit": plan.audit,
            "target": {
                "p_world_m": self.target_position,
                "R_world_from_tool": self.target_rotation,
                "v_world_m_s": self.target_linear_velocity,
                "omega_world_rad_s": self.target_angular_velocity,
                "roll_about_normal_deg": self.roll_about_normal_deg,
                "tool_reference_local_m": self.tool_reference,
            },
        }
