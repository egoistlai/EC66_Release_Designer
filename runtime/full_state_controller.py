"""Interactive EMCP-tool p/R/v/omega controller and emergency-brake launcher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation
import warp as wp

import newton
import newton.examples

from kinematics import FullIKResult
from particle_distribution import DistributionConfig, world_to_bucket
from pose_controller import Example as PoseExample
from trajectory_planning import BucketCollisionModel


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


def _limit_vector(value: np.ndarray, maximum_norm: float) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm > maximum_norm:
        vector = vector * (maximum_norm / norm)
    return vector


class FullStateExample(PoseExample):
    def __init__(self, viewer, args):
        self.full_config_path = Path(args.full_state_config).resolve()
        self.full_cfg = json.loads(self.full_config_path.read_text(encoding="utf-8"))
        self.distribution_config = DistributionConfig.load(
            Path(args.particle_distribution_config).resolve()
        )
        self.show_bucket_frame = bool(args.show_bucket_frame)
        self.bucket_frame_scale_m = float(args.bucket_frame_scale)
        self.bucket_frame_lift_m = float(args.bucket_frame_lift)
        if self.bucket_frame_scale_m <= 0.0:
            raise ValueError("--bucket-frame-scale must be positive")
        if self.bucket_frame_lift_m < 0.0:
            raise ValueError("--bucket-frame-lift must be nonnegative")
        self.qd = np.zeros(6, dtype=float)
        super().__init__(viewer, args)
        self.tool_reference_local = np.asarray(
            self.full_cfg["tool_reference_local_m"], dtype=float
        )
        self.joint_velocity_limits = np.deg2rad(
            np.asarray(self.full_cfg["joint_velocity_limits_deg_s"], dtype=float)
        )
        self.joint_acceleration_limits = np.deg2rad(
            np.asarray(self.full_cfg["joint_acceleration_limits_deg_s2"], dtype=float)
        )
        initial_pose = self.kinematics.tool_pose_world(
            self.initial_q, self.tool_reference_local
        )
        self.target_position = initial_pose[:3, 3].copy()
        self.target_rotation = initial_pose[:3, :3].copy()
        self.target_rpy_deg = Rotation.from_matrix(self.target_rotation).as_euler(
            "xyz", degrees=True
        )
        self.target_linear_velocity = _limit_vector(
            np.asarray(args.initial_linear_velocity, dtype=float),
            float(self.full_cfg["maximum_linear_speed_m_s"]),
        )
        self.target_angular_velocity = _limit_vector(
            np.asarray(args.initial_angular_velocity, dtype=float),
            float(self.full_cfg["maximum_angular_speed_rad_s"]),
        )
        self.full_result: FullIKResult | None = None
        self.actual_twist = np.zeros(6, dtype=float)
        self.twist_linear_error = 0.0
        self.twist_angular_error = 0.0
        self.twist_reachable = False
        self.brake_report: dict | None = None
        self.brake_path = PROJECT / "outputs" / "emergency_brake" / "trajectory.npz"
        self.brake_process: subprocess.Popen | None = None
        self.planner_process: subprocess.Popen | None = None
        planner_cfg = json.loads((HERE / "trajectory_config.json").read_text(encoding="utf-8"))
        self.collision = BucketCollisionModel(self.kinematics, self.cfg, planner_cfg)
        self._solve_full(use_initial_seed=True)

    def _apply_joint_pose(self) -> None:
        self.q_host[self.joint_q_ids] = self.q
        self.qd_host.fill(0.0)
        self.qd_host[self.joint_dof_ids] = getattr(self, "qd", np.zeros(6))
        self.state.joint_q.assign(self.q_host)
        self.state.joint_qd.assign(self.qd_host)
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)

    def _solve_twist(self) -> None:
        desired = np.r_[self.target_linear_velocity, self.target_angular_velocity]
        jacobian = self.kinematics.spatial_jacobian(self.q, self.tool_reference_local)
        linear_tolerance = float(self.full_cfg["twist_linear_tolerance_m_s"])
        angular_tolerance = float(self.full_cfg["twist_angular_tolerance_rad_s"])
        weights = np.diag(
            np.r_[np.full(3, 1.0 / linear_tolerance), np.full(3, 1.0 / angular_tolerance)]
        )
        bounds = (
            float(self.full_cfg["joint_velocity_design_fraction"])
            * self.joint_velocity_limits
        )
        augmented = np.vstack((weights @ jacobian, 0.02 * np.eye(6)))
        target = np.r_[weights @ desired, np.zeros(6)]
        solution = lsq_linear(
            augmented, target, bounds=(-bounds, bounds), lsmr_tol="auto", max_iter=200
        )
        self.qd = np.asarray(solution.x, dtype=float)
        self.actual_twist = jacobian @ self.qd
        self.twist_linear_error = float(
            np.linalg.norm(self.actual_twist[:3] - self.target_linear_velocity)
        )
        self.twist_angular_error = float(
            np.linalg.norm(self.actual_twist[3:] - self.target_angular_velocity)
        )
        self.twist_reachable = bool(
            self.twist_linear_error <= linear_tolerance
            and self.twist_angular_error <= angular_tolerance
            and np.all(np.abs(self.qd) <= bounds + 1.0e-9)
        )

    def _solve_full(self, *, use_initial_seed: bool = False) -> None:
        seed = self.initial_q if use_initial_seed else self.q
        self.target_rotation = Rotation.from_euler(
            "xyz", self.target_rpy_deg, degrees=True
        ).as_matrix()
        self.full_result = self.kinematics.solve_full_pose(
            seed,
            self.target_position,
            self.target_rotation,
            tool_reference_local_m=self.tool_reference_local,
            max_evaluations=int(self.cfg.get("ik_max_evaluations", 180)),
            position_tolerance_m=float(self.full_cfg["ik_position_tolerance_m"]),
            orientation_tolerance_deg=float(self.full_cfg["ik_orientation_tolerance_deg"]),
        )
        self.q = self.full_result.q_rad.copy()
        self._solve_twist()
        self.solve_count += 1
        self.brake_report = None
        self._apply_joint_pose()

    @staticmethod
    def _log_arrow(viewer, name: str, origin: np.ndarray, vector: np.ndarray, color) -> None:
        viewer.log_arrows(
            name,
            wp.array([origin.astype(np.float32)], dtype=wp.vec3),
            wp.array([(origin + vector).astype(np.float32)], dtype=wp.vec3),
            color,
            width=0.009,
        )

    def _render_bucket_local_frame(self) -> None:
        """Draw the distribution/trajectory Bucket-local frame used by this project."""
        config = self.distribution_config
        rotation = config.world_from_bucket_rotation
        exact_origin = config.bucket_center_world_m
        # The mathematical origin is at the bottom centre and would be hidden by
        # the solid floor. Lift only the display glyph; all reported coordinates
        # continue to use exact_origin.
        display_origin = exact_origin + self.bucket_frame_lift_m * rotation[:, 2]
        self.viewer.log_points(
            "/bucket_local/exact_origin",
            wp.array([exact_origin.astype(np.float32)], dtype=wp.vec3),
            0.025,
            wp.array([(1.0, 1.0, 1.0)], dtype=wp.vec3),
        )
        self.viewer.log_points(
            "/bucket_local/display_origin",
            wp.array([display_origin.astype(np.float32)], dtype=wp.vec3),
            0.032,
            wp.array([(1.0, 1.0, 1.0)], dtype=wp.vec3),
        )
        if self.bucket_frame_lift_m > 0.0:
            self.viewer.log_lines(
                "/bucket_local/origin_lift",
                wp.array([exact_origin.astype(np.float32)], dtype=wp.vec3),
                wp.array([display_origin.astype(np.float32)], dtype=wp.vec3),
                (0.85, 0.85, 0.85),
                width=0.006,
            )
        colors = ((1.0, 0.08, 0.05), (0.08, 1.0, 0.16), (0.05, 0.42, 1.0))
        labels = ("X_red", "Y_green", "Z_blue")
        for index, (label, color) in enumerate(zip(labels, colors)):
            axis = rotation[:, index]
            self._log_arrow(
                self.viewer,
                f"/bucket_local/positive_{label}",
                display_origin,
                self.bucket_frame_scale_m * axis,
                color,
            )
            self.viewer.log_lines(
                f"/bucket_local/negative_{label}",
                wp.array([display_origin.astype(np.float32)], dtype=wp.vec3),
                wp.array(
                    [
                        (
                            display_origin
                            - 0.45 * self.bucket_frame_scale_m * axis
                        ).astype(np.float32)
                    ],
                    dtype=wp.vec3,
                ),
                tuple(0.48 * component for component in color),
                width=0.006,
            )

    @staticmethod
    def create_parser():
        parser = PoseExample.create_parser()
        parser.add_argument(
            "--full-state-config", default=str(HERE / "full_state_config.json")
        )
        parser.add_argument(
            "--initial-linear-velocity", type=float, nargs=3, default=(0.0, 0.0, 0.0)
        )
        parser.add_argument(
            "--initial-angular-velocity", type=float, nargs=3, default=(0.0, 0.0, 0.0)
        )
        parser.add_argument(
            "--particle-distribution-config",
            default=str(HERE / "particle_distribution_config.json"),
        )
        parser.add_argument(
            "--show-bucket-frame",
            action=argparse.BooleanOptionalAction,
            default=False,
        )
        parser.add_argument("--bucket-frame-scale", type=float, default=0.55)
        parser.add_argument("--bucket-frame-lift", type=float, default=0.04)
        return parser
