"""Comparable EC66 planners with shared collision, timing and audit semantics."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Callable
import xml.etree.ElementTree as ET

import numpy as np
from scipy.interpolate import PPoly, make_interp_spline
from scipy.optimize import least_squares, lsq_linear, minimize
from scipy.spatial.transform import Rotation
import trimesh

from kinematics import EC66Kinematics, mount_transform, transform


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def _quintic_coefficients(
    duration: float,
    q0: np.ndarray,
    qd0: np.ndarray,
    qdd0: np.ndarray,
    q1: np.ndarray,
    qd1: np.ndarray,
    qdd1: np.ndarray,
) -> np.ndarray:
    t = float(duration)
    matrix = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0],
            [1, t, t**2, t**3, t**4, t**5],
            [0, 1, 2 * t, 3 * t**2, 4 * t**3, 5 * t**4],
            [0, 0, 2, 6 * t, 12 * t**2, 20 * t**3],
        ],
        dtype=float,
    )
    boundary = np.vstack((q0, qd0, qdd0, q1, qd1, qdd1))
    return np.linalg.solve(matrix, boundary)[::-1]


def quintic_motion(q_start: np.ndarray, q_goal: np.ndarray, qd_goal: np.ndarray, duration: float) -> PPoly:
    coefficients = _quintic_coefficients(
        duration,
        q_start,
        np.zeros(6),
        np.zeros(6),
        q_goal,
        qd_goal,
        np.zeros(6),
    )
    return PPoly(coefficients[:, None, :], [0.0, float(duration)])


def quintic_boundary_motion(
    q_start: np.ndarray,
    qd_start: np.ndarray,
    qdd_start: np.ndarray,
    q_goal: np.ndarray,
    qd_goal: np.ndarray,
    qdd_goal: np.ndarray,
    duration: float,
) -> PPoly:
    """Quintic segment with explicit position/velocity/acceleration boundaries."""
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("Quintic duration must be finite and positive")
    coefficients = _quintic_coefficients(
        float(duration),
        np.asarray(q_start, dtype=float).reshape(6),
        np.asarray(qd_start, dtype=float).reshape(6),
        np.asarray(qdd_start, dtype=float).reshape(6),
        np.asarray(q_goal, dtype=float).reshape(6),
        np.asarray(qd_goal, dtype=float).reshape(6),
        np.asarray(qdd_goal, dtype=float).reshape(6),
    )
    return PPoly(coefficients[:, None, :], [0.0, float(duration)])


def _ppoly_from_spline(spline, start: float, stop: float) -> PPoly:
    curves = [PPoly.from_spline((spline.t, spline.c[:, joint], spline.k)) for joint in range(6)]
    knots = np.unique(np.r_[start, curves[0].x[(curves[0].x > start) & (curves[0].x < stop)], stop])
    degree = spline.k
    coefficients = np.zeros((degree + 1, len(knots) - 1, 6), dtype=float)
    for interval, left in enumerate(knots[:-1]):
        for derivative in range(degree + 1):
            coefficients[degree - derivative, interval] = np.asarray(
                [curve(left, derivative) for curve in curves]
            ) / math.factorial(derivative)
    return PPoly(coefficients, knots)


def spline_motion(waypoints: np.ndarray, qd_goal: np.ndarray, duration: float) -> PPoly:
    values = np.asarray(waypoints, dtype=float)
    if len(values) < 3:
        values = np.vstack((values[0], 0.5 * (values[0] + values[-1]), values[-1]))
    chord = np.linalg.norm(np.diff(values, axis=0), axis=1)
    chord = np.maximum(chord, 1.0e-5)
    times = np.r_[0.0, np.cumsum(chord)]
    times *= float(duration) / times[-1]
    spline = make_interp_spline(
        times,
        values,
        k=min(5, len(values) + 2),
        bc_type=(
            [(1, np.zeros(6)), (2, np.zeros(6))],
            [(1, np.asarray(qd_goal, dtype=float)), (2, np.zeros(6))],
        ),
    )
    return _ppoly_from_spline(spline, 0.0, float(duration))


class BucketCollisionModel:
    """Fast conservative pot clearance plus URDF-based robot self-collision."""

    def __init__(self, kinematics: EC66Kinematics, scene: dict, planner: dict):
        self.kinematics = kinematics
        self.center_xy = np.asarray(scene["bucket_xyz"][:2], dtype=float)
        self.bottom_z = float(scene["bucket_xyz"][2])
        self.radius = float(scene["bucket_radius"])
        self.height = float(scene["bucket_height"])
        self.wall = float(scene["bucket_wall"])
        self.samples_per_link = int(planner["collision_samples_per_link"])
        self.required_clearance = float(planner["collision_clearance_m"])
        self.required_self_clearance = float(planner.get("self_collision_clearance_m", 0.0))

        raw = trimesh.load(HERE / scene["shovel_mesh"], force="mesh")
        vertices = (
            np.asarray(raw.vertices) - np.asarray(scene["shovel_origin_stl"], dtype=float)
        ) * float(scene["shovel_scale"])
        vertices = vertices[:, [2, 0, 1]]
        sample_count = min(160, len(vertices))
        indices = np.linspace(0, len(vertices) - 1, sample_count, dtype=int)
        self.shovel_points_local = vertices[indices]
        self.shovel_box_center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
        self.shovel_box_half = 0.5 * (vertices.max(axis=0) - vertices.min(axis=0))

        urdf_root = ET.parse(self.kinematics.urdf_path).getroot()
        self.arm_boxes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for link in urdf_root.findall("link"):
            name = link.get("name")
            if name not in {"base_link", "link1", "link2", "link3", "link4", "link5", "link6"}:
                continue
            mesh_element = link.find("collision/geometry/mesh")
            if mesh_element is None:
                continue
            mesh_path = self.kinematics.urdf_path.parent / mesh_element.get("filename")
            mesh = trimesh.load(mesh_path, force="mesh")
            bounds = np.asarray(mesh.bounds, dtype=float)
            self.arm_boxes[name] = (0.5 * bounds.sum(axis=0), 0.5 * (bounds[1] - bounds[0]))
        self.arm_box_order = tuple(
            name for name in ("base_link", "link1", "link2", "link3", "link4", "link5", "link6")
            if name in self.arm_boxes
        )

    @staticmethod
    def _rectangle_sdf(x: np.ndarray, center: np.ndarray, half: np.ndarray) -> np.ndarray:
        delta = np.abs(x - center) - half
        return np.linalg.norm(np.maximum(delta, 0.0), axis=-1) + np.minimum(
            np.maximum(delta[..., 0], delta[..., 1]), 0.0
        )

    def point_clearance(self, points_world: np.ndarray) -> np.ndarray:
        points = np.asarray(points_world, dtype=float)
        radial = np.linalg.norm(points[:, :2] - self.center_xy, axis=1)
        rz = np.column_stack((radial, points[:, 2]))
        wall_center = np.array([self.radius + 0.5 * self.wall, self.bottom_z + 0.5 * self.height])
        wall_half = np.array([0.5 * self.wall, 0.5 * self.height])
        wall_sdf = self._rectangle_sdf(rz, wall_center, wall_half)

        floor_center = np.array([0.0, self.bottom_z + 0.5 * self.wall])
        floor_delta = np.column_stack(
            (radial - (self.radius + self.wall), np.abs(points[:, 2] - floor_center[1]) - 0.5 * self.wall)
        )
        floor_sdf = np.linalg.norm(np.maximum(floor_delta, 0.0), axis=1) + np.minimum(
            np.maximum(floor_delta[:, 0], floor_delta[:, 1]), 0.0
        )
        return np.minimum(wall_sdf, floor_sdf)

    def min_clearance(self, q_rad: np.ndarray) -> float:
        chain = self.kinematics.chain_points_world(q_rad)
        link_samples = []
        link_radii = []
        # Conservative capsules around the arm centreline. Duplicate URDF
        # origins are skipped; radii roughly cover the supplied EC66 meshes.
        for index, (start, end) in enumerate(zip(chain[:-1], chain[1:])):
            if np.linalg.norm(end - start) < 1.0e-8:
                continue
            u = np.linspace(0.0, 1.0, self.samples_per_link)
            link_samples.append(start[None, :] + u[:, None] * (end - start)[None, :])
            radius = 0.085 if index < 3 else 0.065
            link_radii.extend([radius] * len(u))
        link_points = np.vstack(link_samples)
        link_clearance = self.point_clearance(link_points) - np.asarray(link_radii)

        shovel_pose = self.kinematics.shovel_pose_world(q_rad)
        shovel_points = self.shovel_points_local @ shovel_pose[:3, :3].T + shovel_pose[:3, 3]
        shovel_clearance = self.point_clearance(shovel_points) - 0.004
        return float(min(np.min(link_clearance), np.min(shovel_clearance)))

    @staticmethod
    def _obb_axis_clearance(
        center_a: np.ndarray,
        axes_a: np.ndarray,
        half_a: np.ndarray,
        center_b: np.ndarray,
        axes_b: np.ndarray,
        half_b: np.ndarray,
    ) -> float:
        """Largest separating-axis gap; negative values mean OBB penetration."""
        delta = center_b - center_a
        cross_axes = np.asarray(
            [np.cross(axes_a[:, i], axes_b[:, j]) for i in range(3) for j in range(3)]
        )
        axes = np.vstack((axes_a.T, axes_b.T, cross_axes))
        norms = np.linalg.norm(axes, axis=1)
        unit = axes[norms >= 1.0e-10] / norms[norms >= 1.0e-10, None]
        radius_a = np.abs(unit @ axes_a) @ half_a
        radius_b = np.abs(unit @ axes_b) @ half_b
        gaps = np.abs(unit @ delta) - radius_a - radius_b
        return float(np.max(gaps))

    @staticmethod
    def _world_box(
        pose: np.ndarray, local_center: np.ndarray, local_half: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rotation = np.asarray(pose[:3, :3], dtype=float)
        center = rotation @ np.asarray(local_center, dtype=float) + pose[:3, 3]
        return center, rotation, np.asarray(local_half, dtype=float)

    def self_clearance(self, q_rad: np.ndarray) -> tuple[float, str]:
        """Conservative non-adjacent-link and shovel-to-arm OBB clearance."""
        poses = self.kinematics.link_poses_world(q_rad)
        boxes = {
            name: self._world_box(poses[name], *self.arm_boxes[name])
            for name in self.arm_box_order
        }
        minimum = np.inf
        pair = "none"
        for left_index, left_name in enumerate(self.arm_box_order):
            # Two-link neighborhoods share bulky joint housings; their local
            # mesh AABBs intentionally overlap even in nominal configurations.
            for right_index in range(left_index + 3, len(self.arm_box_order)):
                right_name = self.arm_box_order[right_index]
                gap = self._obb_axis_clearance(*boxes[left_name], *boxes[right_name])
                if gap < minimum:
                    minimum, pair = gap, f"{left_name}:{right_name}"

        shovel_pose = self.kinematics.shovel_pose_world(q_rad)
        shovel_box = self._world_box(
            shovel_pose, self.shovel_box_center, self.shovel_box_half
        )
        # link5/link6 are the wrist attachment neighborhood and intentionally
        # overlap the shovel mounting envelope. All earlier links are checked.
        for name in self.arm_box_order:
            if name in {"link5", "link6"}:
                continue
            gap = self._obb_axis_clearance(*shovel_box, *boxes[name])
            if gap < minimum:
                minimum, pair = gap, f"shovel:{name}"
        return float(minimum), pair

    def state_valid(self, q_rad: np.ndarray) -> bool:
        q = np.asarray(q_rad, dtype=float)
        self_clearance, _ = self.self_clearance(q)
        return bool(
            np.all(q >= self.kinematics.low)
            and np.all(q <= self.kinematics.high)
            and self.min_clearance(q) >= self.required_clearance
            and self_clearance >= self.required_self_clearance
        )

    def edge_valid(self, q0: np.ndarray, q1: np.ndarray, joint_step_rad: float = 0.06) -> bool:
        count = max(2, int(np.ceil(np.max(np.abs(q1 - q0)) / joint_step_rad)) + 1)
        return all(self.state_valid((1.0 - u) * q0 + u * q1) for u in np.linspace(0.0, 1.0, count))


@dataclass
class Plan:
    planner_id: str
    label: str
    family: str
    motion: PPoly
    waypoints: np.ndarray
    planning_seconds: float
    audit: dict


class PlanningContext:
    def __init__(
        self,
        terminal_speed_m_s: float | None = None,
        target_position_world_m: np.ndarray | None = None,
        target_normal_world: np.ndarray | None = None,
    ):
        self.scene = load_json(HERE / "scene_config.json")
        self.settings = load_json(HERE / "trajectory_config.json")
        if terminal_speed_m_s is not None:
            self.settings["target_terminal_speed_m_s"] = float(terminal_speed_m_s)
        if target_position_world_m is not None:
            self.settings["target_position_world_m"] = np.asarray(
                target_position_world_m, dtype=float
            ).tolist()
        if target_normal_world is not None:
            normal = np.asarray(target_normal_world, dtype=float)
            if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) < 1.0e-9:
                raise ValueError("target_normal_world must be a finite nonzero 3-vector")
            self.settings["target_normal_world"] = (normal / np.linalg.norm(normal)).tolist()
        self.q_start = np.deg2rad(np.asarray(self.settings["q_start_deg"], dtype=float))
        self.target_position = np.asarray(self.settings["target_position_world_m"], dtype=float)
        self.target_normal = np.asarray(self.settings["target_normal_world"], dtype=float)
        self.target_normal /= np.linalg.norm(self.target_normal)
        self.joint_velocity_limits = np.deg2rad(
            np.asarray(self.settings["joint_velocity_limits_deg_s"], dtype=float)
        )
        self.joint_acceleration_limits = np.deg2rad(
            np.asarray(self.settings["joint_acceleration_limits_deg_s2"], dtype=float)
        )
        world_from_base = transform(
            self.scene["world_from_base_xyz"], np.deg2rad(self.scene["world_from_base_rpy_deg"])
        )
        mount = mount_transform(self.scene["mount_rotations_deg"], self.scene["mount_xyz"])
        self.kinematics = EC66Kinematics(
            HERE / self.scene["urdf"],
            world_from_base,
            mount,
            self.scene["shovel_normal_local"],
        )
        self.collision = BucketCollisionModel(self.kinematics, self.scene, self.settings)
        self.rng = np.random.default_rng(int(self.settings["random_seed"]))
        self.goal_ik = self.kinematics.solve(
            self.q_start,
            self.target_position,
            self.target_normal,
            max_evaluations=250,
            position_tolerance_m=float(self.settings["goal_position_tolerance_m"]),
            normal_tolerance_deg=float(self.settings["goal_normal_tolerance_deg"]),
        )
        self.q_goal = self.goal_ik.q_rad.copy()
        self.qd_goal, self.terminal_velocity_fit = self._solve_terminal_velocity()

    def _solve_terminal_velocity(self) -> tuple[np.ndarray, dict]:
        jacobian = self.kinematics.position_jacobian(self.q_goal)
        desired = self.goal_ik.normal_world * float(self.settings["target_terminal_speed_m_s"])
        design = float(self.settings["design_limit_fraction"])
        bounds = design * self.joint_velocity_limits
        augmented = np.vstack((jacobian, 0.015 * np.eye(6)))
        target = np.r_[desired, np.zeros(6)]
        solution = lsq_linear(augmented, target, bounds=(-bounds, bounds), lsmr_tol="auto")
        achieved = jacobian @ solution.x
        speed = float(np.linalg.norm(achieved))
        cosine = float(np.dot(achieved, self.goal_ik.normal_world) / max(speed, 1.0e-12))
        return np.asarray(solution.x, dtype=float), {
            "requested_world_m_s": desired,
            "achieved_world_m_s": achieved,
            "requested_speed_m_s": float(np.linalg.norm(desired)),
            "achieved_speed_m_s": speed,
            "direction_error_deg": float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))),
            "joint_velocity_deg_s": np.rad2deg(solution.x),
        }

    def audit_motion(self, motion: PPoly) -> dict:
        start, stop = float(motion.x[0]), float(motion.x[-1])
        step = float(self.settings["trajectory_sample_step_s"])
        count = max(80, int(np.ceil((stop - start) / step)) + 1)
        times = np.linspace(start, stop, count)
        q = np.asarray(motion(times), dtype=float)
        qd = np.asarray(motion(times, 1), dtype=float)
        qdd = np.asarray(motion(times, 2), dtype=float)
        positions = []
        normals = []
        rotations = []
        clearances = []
        self_clearance_records = []
        for angles in q:
            pose = self.kinematics.shovel_pose_world(angles)
            positions.append(pose[:3, 3])
            normals.append(pose[:3, :3] @ self.kinematics.shovel_normal_local)
            rotations.append(pose[:3, :3])
            clearances.append(self.collision.min_clearance(angles))
            self_clearance_records.append(self.collision.self_clearance(angles))
        positions = np.asarray(positions)
        normals = np.asarray(normals)
        rotations = np.asarray(rotations)
        sample_dt = times[1] - times[0]
        linear_velocity = np.gradient(positions, sample_dt, axis=0, edge_order=2)
        linear_acceleration = np.gradient(linear_velocity, sample_dt, axis=0, edge_order=2)
        angular_velocity = np.asarray(
            [
                Rotation.from_matrix(rotations[index + 1] @ rotations[index].T).as_rotvec() / sample_dt
                for index in range(len(rotations) - 1)
            ]
        )
        angular_acceleration = np.gradient(angular_velocity, sample_dt, axis=0, edge_order=2)
        joint_velocity_ratio = float(np.max(np.abs(qd) / self.joint_velocity_limits))
        joint_acceleration_ratio = float(np.max(np.abs(qdd) / self.joint_acceleration_limits))
        tcp_ratios = {
            "linear_velocity": float(
                np.max(np.linalg.norm(linear_velocity, axis=1))
                / float(self.settings["tcp_linear_velocity_limit_m_s"])
            ),
            "linear_acceleration": float(
                np.max(np.linalg.norm(linear_acceleration, axis=1))
                / float(self.settings["tcp_linear_acceleration_limit_m_s2"])
            ),
            "angular_velocity": float(
                np.max(np.linalg.norm(angular_velocity, axis=1))
                / np.deg2rad(float(self.settings["tcp_angular_velocity_limit_deg_s"]))
            ),
            "angular_acceleration": float(
                np.max(np.linalg.norm(angular_acceleration, axis=1))
                / np.deg2rad(float(self.settings["tcp_angular_acceleration_limit_deg_s2"]))
            ),
        }
        terminal_velocity = self.kinematics.position_jacobian(q[-1]) @ qd[-1]
        terminal_speed = float(np.linalg.norm(terminal_velocity))
        direction_cosine = float(
            np.dot(terminal_velocity, normals[-1]) / max(terminal_speed, 1.0e-12)
        )
        path_segments = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        path_length = float(np.sum(path_segments))
        apparent_force = np.array([0.0, 0.0, -9.81]) - linear_acceleration
        support = -np.sum(apparent_force * normals, axis=1)
        tangent = np.linalg.norm(apparent_force + support[:, None] * normals, axis=1)
        centroid_force_violation = float(
            np.mean(np.maximum(tangent - 0.4 * support + 0.3, 0.0))
            + np.mean(np.maximum(0.5 - support, 0.0))
        )
        emcp_prefilter_score = float(
            centroid_force_violation + 0.15 * path_length + 0.03 * (stop - start)
        )
        displacement = float(np.linalg.norm(positions[-1] - positions[0]))
        path_detour = path_length / max(displacement, 0.02)
        joint_total_travel_deg = np.rad2deg(np.sum(np.abs(np.diff(q, axis=0)), axis=0))
        workspace_min = np.asarray(self.settings["workspace_min_world_m"], dtype=float)
        workspace_max = np.asarray(self.settings["workspace_max_world_m"], dtype=float)
        workspace_valid = bool(np.all(positions >= workspace_min) and np.all(positions <= workspace_max))
        spatial_valid = bool(
            workspace_valid
            and path_length <= float(self.settings["max_tcp_path_length_m"])
            and path_detour <= float(self.settings["max_tcp_detour_ratio"])
            and np.all(
                joint_total_travel_deg
                <= np.asarray(self.settings["max_joint_total_travel_deg"], dtype=float)
            )
        )
        position_error = float(np.linalg.norm(positions[-1] - self.target_position))
        normal_error = float(
            np.degrees(np.arccos(np.clip(np.dot(normals[-1], self.target_normal), -1.0, 1.0)))
        )
        hard = float(self.settings["hard_limit_fraction"])
        minimum_clearance = float(np.min(clearances))
        self_clearances = np.asarray([record[0] for record in self_clearance_records])
        minimum_self_index = int(np.argmin(self_clearances))
        minimum_self_clearance = float(self_clearances[minimum_self_index])
        valid = bool(
            np.all(q >= self.kinematics.low - 1.0e-9)
            and np.all(q <= self.kinematics.high + 1.0e-9)
            and joint_velocity_ratio <= hard + 1.0e-3
            and joint_acceleration_ratio <= hard + 1.0e-3
            and max(tcp_ratios.values()) <= hard + 5.0e-3
            and minimum_clearance >= float(self.settings["collision_clearance_m"])
            and minimum_self_clearance
            >= float(self.settings.get("self_collision_clearance_m", 0.0))
            and position_error <= float(self.settings["goal_position_tolerance_m"])
            and normal_error <= float(self.settings["goal_normal_tolerance_deg"])
            and spatial_valid
        )
        return json_safe(
            {
                "valid": valid,
                "duration_s": stop - start,
                "sample_count": count,
                "joint_velocity_ratio": joint_velocity_ratio,
                "joint_acceleration_ratio": joint_acceleration_ratio,
                "tcp_ratios": tcp_ratios,
                "tcp_max_ratio": max(tcp_ratios.values()),
                "minimum_bucket_clearance_m": minimum_clearance,
                "minimum_self_clearance_m": minimum_self_clearance,
                "minimum_self_collision_pair": self_clearance_records[minimum_self_index][1],
                "bucket_collision_free": minimum_clearance >= float(self.settings["collision_clearance_m"]),
                "self_collision_free": minimum_self_clearance
                >= float(self.settings.get("self_collision_clearance_m", 0.0)),
                "collision_free": bool(
                    minimum_clearance >= float(self.settings["collision_clearance_m"])
                    and minimum_self_clearance
                    >= float(self.settings.get("self_collision_clearance_m", 0.0))
                ),
                "collision_scope": "sampled EC66/shovel versus pot plus non-adjacent URDF-link and shovel-to-arm OBB self-collision",
                "path_length_m": path_length,
                "emcp_nominal_centroid_force_violation": centroid_force_violation,
                "emcp_inspired_prefilter_score": emcp_prefilter_score,
                "path_detour_ratio": path_detour,
                "joint_total_travel_deg": joint_total_travel_deg,
                "workspace_valid": workspace_valid,
                "spatial_valid": spatial_valid,
                "goal_position_error_m": position_error,
                "goal_normal_error_deg": normal_error,
                "terminal_velocity_world_m_s": terminal_velocity,
                "terminal_speed_m_s": terminal_speed,
                "terminal_velocity_direction_error_deg": float(
                    np.degrees(np.arccos(np.clip(direction_cosine, -1.0, 1.0)))
                ),
                "start_velocity_max_deg_s": float(np.max(np.abs(np.rad2deg(qd[0])))),
                "start_acceleration_max_deg_s2": float(np.max(np.abs(np.rad2deg(qdd[0])))),
            }
        )

    def _retime(self, builder: Callable[[float], PPoly], initial_duration: float = 1.5) -> tuple[PPoly, dict]:
        duration = float(initial_duration)
        design = float(self.settings["design_limit_fraction"])
        last_audit = None
        for _ in range(28):
            motion = builder(duration)
            last_audit = self.audit_motion(motion)
            dynamic_ratio = max(
                last_audit["joint_velocity_ratio"],
                math.sqrt(max(0.0, last_audit["joint_acceleration_ratio"])),
                last_audit["tcp_ratios"]["linear_velocity"],
                math.sqrt(max(0.0, last_audit["tcp_ratios"]["linear_acceleration"])),
                last_audit["tcp_ratios"]["angular_velocity"],
                math.sqrt(max(0.0, last_audit["tcp_ratios"]["angular_acceleration"])),
            )
            if dynamic_ratio <= design + 2.0e-3:
                return motion, last_audit
            duration *= max(1.10, min(1.45, dynamic_ratio / design))
            if duration > 14.0:
                break
        return motion, last_audit

    def _make_plan(self, planner_id: str, label: str, family: str, waypoints: np.ndarray, started: float) -> Plan:
        values = np.asarray(waypoints, dtype=float)
        if len(values) <= 2:
            builder = lambda duration: quintic_motion(self.q_start, self.q_goal, self.qd_goal, duration)
        else:
            builder = lambda duration: spline_motion(values, self.qd_goal, duration)
        motion, audit = self._retime(builder)
        return Plan(
            planner_id=planner_id,
            label=label,
            family=family,
            motion=motion,
            waypoints=values,
            planning_seconds=time.perf_counter() - started,
            audit=audit,
        )

    def plan_minimum_jerk(self) -> Plan:
        started = time.perf_counter()
        return self._make_plan(
            "minimum_jerk",
            "Joint-space minimum jerk",
            "direct polynomial",
            np.vstack((self.q_start, self.q_goal)),
            started,
        )

    @staticmethod
    def _spherical_normal(start: np.ndarray, goal: np.ndarray, u: float) -> np.ndarray:
        cosine = float(np.clip(np.dot(start, goal), -1.0, 1.0))
        angle = float(np.arccos(cosine))
        if angle < 1.0e-7:
            return start.copy()
        return (np.sin((1.0 - u) * angle) * start + np.sin(u * angle) * goal) / np.sin(angle)

    def plan_cartesian_ik(self, waypoint_count: int = 13) -> Plan:
        started = time.perf_counter()
        start_position, start_normal = self.kinematics.position_and_normal(self.q_start)
        waypoints = [self.q_start.copy()]
        previous = self.q_start.copy()
        for u in np.linspace(0.0, 1.0, waypoint_count)[1:-1]:
            smooth_u = 3.0 * u**2 - 2.0 * u**3
            position = (1.0 - smooth_u) * start_position + smooth_u * self.target_position
            position[2] += 0.10 * np.sin(np.pi * u)
            normal = self._spherical_normal(start_normal, self.target_normal, smooth_u)
            result = self.kinematics.solve(
                previous,
                position,
                normal,
                max_evaluations=120,
                position_tolerance_m=0.012,
                normal_tolerance_deg=8.0,
            )
            previous = result.q_rad
            waypoints.append(previous.copy())
        waypoints.append(self.q_goal.copy())
        return self._make_plan(
            "cartesian_ik",
            "Cartesian IK continuation",
            "task-space continuation",
            np.asarray(waypoints),
            started,
        )

    @staticmethod
    def _nearest(tree: list[np.ndarray], sample: np.ndarray) -> int:
        return int(np.argmin([np.linalg.norm(node - sample) for node in tree]))

    def plan_rrt_connect(self, max_iterations: int = 1800, step_rad: float = 0.28) -> Plan:
        started = time.perf_counter()
        if self.collision.edge_valid(self.q_start, self.q_goal):
            path = [self.q_start.copy(), self.q_goal.copy()]
        else:
            trees = [[self.q_start.copy()], [self.q_goal.copy()]]
            parents = [[-1], [-1]]
            connected = None
            lower = np.maximum(self.kinematics.low, np.minimum(self.q_start, self.q_goal) - 1.2)
            upper = np.minimum(self.kinematics.high, np.maximum(self.q_start, self.q_goal) + 1.2)
            for iteration in range(max_iterations):
                active = iteration % 2
                other = 1 - active
                sample = (
                    trees[other][0]
                    if self.rng.random() < 0.18
                    else self.rng.uniform(lower, upper)
                )
                nearest = self._nearest(trees[active], sample)
                delta = sample - trees[active][nearest]
                candidate = trees[active][nearest] + delta * min(1.0, step_rad / max(np.linalg.norm(delta), 1.0e-12))
                if not self.collision.edge_valid(trees[active][nearest], candidate):
                    continue
                trees[active].append(candidate)
                parents[active].append(nearest)
                new_index = len(trees[active]) - 1
                other_index = self._nearest(trees[other], candidate)
                if self.collision.edge_valid(candidate, trees[other][other_index]):
                    connected = (active, new_index, other_index)
                    break
            if connected is None:
                raise RuntimeError("RRT-Connect did not find a collision-free path")

            def branch(tree_index: int, node_index: int) -> list[np.ndarray]:
                result = []
                while node_index >= 0:
                    result.append(trees[tree_index][node_index])
                    node_index = parents[tree_index][node_index]
                return result[::-1]

            active, active_index, other_index = connected
            active_branch = branch(active, active_index)
            other_branch = branch(1 - active, other_index)
            path = active_branch + other_branch[::-1]
            if active == 1:
                path = path[::-1]

            for _ in range(160):
                if len(path) <= 2:
                    break
                i, j = sorted(self.rng.integers(0, len(path), size=2))
                if j > i + 1 and self.collision.edge_valid(path[i], path[j]):
                    path = path[: i + 1] + path[j:]
        return self._make_plan(
            "rrt_connect",
            "Bidirectional RRT-Connect",
            "sampling based",
            np.asarray(path),
            started,
        )

    def plan_penalty_optimization(self, waypoint_count: int = 11) -> Plan:
        started = time.perf_counter()
        initial = np.asarray(
            [(1.0 - u) * self.q_start + u * self.q_goal for u in np.linspace(0.0, 1.0, waypoint_count)]
        )
        lower = np.tile(self.kinematics.low, waypoint_count - 2)
        upper = np.tile(self.kinematics.high, waypoint_count - 2)
        clearance_target = float(self.settings["collision_clearance_m"]) + 0.025

        def unpack(flat: np.ndarray) -> np.ndarray:
            return np.vstack((self.q_start, flat.reshape(-1, 6), self.q_goal))

        def objective(flat: np.ndarray) -> float:
            points = unpack(flat)
            velocity = np.diff(points, axis=0)
            acceleration = np.diff(points, n=2, axis=0)
            smoothness = 2.0 * np.sum(acceleration**2) + 0.04 * np.sum(velocity**2)
            clearance = np.asarray([self.collision.min_clearance(point) for point in points[1:-1]])
            obstacle = 650.0 * np.sum(np.maximum(0.0, clearance_target - clearance) ** 2)
            return float(smoothness + obstacle)

        optimized = minimize(
            objective,
            initial[1:-1].ravel(),
            method="L-BFGS-B",
            bounds=list(zip(lower, upper)),
            options={"maxiter": 90, "ftol": 1.0e-10, "maxls": 25},
        )
        waypoints = unpack(optimized.x)
        plan = self._make_plan(
            "penalty_optimization",
            "Collision-penalty trajectory optimization",
            "gradient optimization",
            waypoints,
            started,
        )
        plan.audit["optimizer"] = {
            "success": bool(optimized.success),
            "message": str(optimized.message),
            "iterations": int(optimized.nit),
            "objective": float(optimized.fun),
        }
        return plan

    def generate_all(self) -> list[Plan]:
        planners = (
            self.plan_minimum_jerk,
            self.plan_cartesian_ik,
            self.plan_rrt_connect,
            self.plan_penalty_optimization,
        )
        plans = []
        for planner in planners:
            plan = planner()
            plans.append(plan)
            print(
                f"{plan.planner_id}: valid={plan.audit['valid']} "
                f"duration={plan.audit['duration_s']:.3f}s "
                f"clearance={plan.audit['minimum_bucket_clearance_m']:.3f}m "
                f"joint={plan.audit['joint_velocity_ratio']:.3f}/{plan.audit['joint_acceleration_ratio']:.3f} "
                f"tcp={plan.audit['tcp_max_ratio']:.3f}",
                flush=True,
            )
        return plans

    def save_plan(self, plan: Plan, output_dir: Path) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{plan.planner_id}.npz"
        duration = float(plan.motion.x[-1])
        sample_times = np.linspace(0.0, duration, max(2, int(np.ceil(duration * 60.0)) + 1))
        q = np.asarray(plan.motion(sample_times), dtype=float)
        poses = []
        for angles in q:
            flange = self.kinematics.flange_pose_base(angles)
            poses.append(np.r_[flange[:3, 3], Rotation.from_matrix(flange[:3, :3]).as_quat()])
        metadata = {
            "planner_id": plan.planner_id,
            "label": plan.label,
            "family": plan.family,
            "planning_seconds": plan.planning_seconds,
            "audit": plan.audit,
            "target": {
                "position_world_m": self.target_position,
                "normal_world": self.target_normal,
                "terminal_speed_m_s": self.settings["target_terminal_speed_m_s"],
            },
        }
        np.savez_compressed(
            path,
            coefficients=plan.motion.c,
            t=plan.motion.x,
            q=q,
            poses=np.asarray(poses),
            errors=np.zeros((len(q), 2), dtype=float),
            release_start_s=duration,
            release_action_start_s=duration,
            release_state_s=duration,
            terminal_qd_rad_s=np.asarray(plan.motion(duration, 1), dtype=float),
            target_position_world_m=self.target_position,
            target_normal_world=self.target_normal,
            metadata_json=json.dumps(json_safe(metadata), ensure_ascii=False),
        )
        return {
            "id": plan.planner_id,
            "label": plan.label,
            "family": plan.family,
            "path": str(path.resolve()),
            "planning_seconds": plan.planning_seconds,
            "audit": plan.audit,
        }
