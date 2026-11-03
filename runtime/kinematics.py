"""EC66 forward/inverse kinematics for position + shovel-normal control."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def transform(xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)) -> np.ndarray:
    result = np.eye(4, dtype=float)
    result[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    result[:3, 3] = np.asarray(xyz, dtype=float)
    return result


def mount_transform(rotations_deg: list[dict], xyz=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Match the fixed-flange-axis convention used by the sibling projects."""
    result = transform(xyz)
    axes = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
    }
    for item in rotations_deg:
        axis = item["axis"].lower()
        if axis not in axes:
            raise ValueError(f"Unknown mount rotation axis: {axis!r}")
        fixed_rotation = Rotation.from_rotvec(
            axes[axis] * np.deg2rad(float(item["angle_deg"]))
        ).as_matrix()
        result[:3, :3] = fixed_rotation @ result[:3, :3]
    return result


def normal_from_angles(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Unit world vector from azimuth in XY and elevation above XY."""
    azimuth = np.deg2rad(float(azimuth_deg))
    elevation = np.deg2rad(float(elevation_deg))
    cos_elevation = np.cos(elevation)
    return np.array(
        [
            cos_elevation * np.cos(azimuth),
            cos_elevation * np.sin(azimuth),
            np.sin(elevation),
        ],
        dtype=float,
    )


def angles_from_normal(normal: np.ndarray) -> tuple[float, float]:
    value = np.asarray(normal, dtype=float).reshape(3)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm < 1.0e-12:
        raise ValueError("Normal must be a finite, non-zero vector")
    value /= norm
    azimuth = np.degrees(np.arctan2(value[1], value[0]))
    elevation = np.degrees(np.arctan2(value[2], np.linalg.norm(value[:2])))
    return float(azimuth), float(elevation)


@dataclass(frozen=True)
class IKResult:
    q_rad: np.ndarray
    position_world_m: np.ndarray
    normal_world: np.ndarray
    position_error_m: float
    normal_error_deg: float
    optimizer_success: bool
    reachable: bool
    cost: float
    evaluations: int


@dataclass(frozen=True)
class FullIKResult:
    q_rad: np.ndarray
    position_world_m: np.ndarray
    rotation_world: np.ndarray
    position_error_m: float
    orientation_error_deg: float
    optimizer_success: bool
    reachable: bool
    cost: float
    evaluations: int


class EC66Kinematics:
    """URDF kinematics with the configured shovel rigidly mounted to the flange."""

    def __init__(
        self,
        urdf_path: str | Path,
        world_from_base: np.ndarray,
        flange_from_shovel: np.ndarray,
        shovel_normal_local=(0.0, 0.0, 1.0),
    ):
        self.urdf_path = Path(urdf_path)
        root = ET.parse(self.urdf_path).getroot()
        self.chain: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]] = []
        self.chain_child_links: list[str] = []
        lower: list[float] = []
        upper: list[float] = []
        names: list[str] = []

        for joint in root.findall("joint"):
            origin = joint.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            moving = joint.get("type") in {"revolute", "continuous"}
            axis_element = joint.find("axis")
            axis = (
                np.fromstring(axis_element.get("xyz"), sep=" ")
                if moving and axis_element is not None
                else None
            )
            self.chain.append((xyz, Rotation.from_euler("xyz", rpy).as_matrix(), axis))
            self.chain_child_links.append(joint.find("child").get("link"))
            if moving:
                limit = joint.find("limit")
                if limit is None:
                    raise ValueError(f"Moving joint {joint.get('name')} has no limits")
                names.append(joint.get("name"))
                lower.append(float(limit.get("lower")))
                upper.append(float(limit.get("upper")))

        if len(lower) != 6:
            raise ValueError(f"Expected six EC66 revolute joints, found {len(lower)}")
        self.joint_names = tuple(names)
        self.low = np.asarray(lower, dtype=float)
        self.high = np.asarray(upper, dtype=float)
        self.world_from_base = np.asarray(world_from_base, dtype=float).reshape(4, 4)
        self.flange_from_shovel = np.asarray(flange_from_shovel, dtype=float).reshape(4, 4)
        local_normal = np.asarray(shovel_normal_local, dtype=float).reshape(3)
        local_normal_norm = np.linalg.norm(local_normal)
        if local_normal_norm < 1.0e-12:
            raise ValueError("shovel_normal_local must be non-zero")
        self.shovel_normal_local = local_normal / local_normal_norm

    def flange_pose_base(self, q_rad: np.ndarray) -> np.ndarray:
        q = np.asarray(q_rad, dtype=float).reshape(6)
        position = np.zeros(3, dtype=float)
        rotation = np.eye(3, dtype=float)
        moving_index = 0
        for offset, origin_rotation, axis in self.chain:
            position = position + rotation @ offset
            rotation = rotation @ origin_rotation
            if axis is not None:
                rotation = rotation @ Rotation.from_rotvec(axis * q[moving_index]).as_matrix()
                moving_index += 1
        result = np.eye(4, dtype=float)
        result[:3, :3] = rotation
        result[:3, 3] = position
        return result

    def chain_points_world(self, q_rad: np.ndarray) -> np.ndarray:
        """World positions along the URDF chain for conservative capsule checks."""
        q = np.asarray(q_rad, dtype=float).reshape(6)
        position = np.zeros(3, dtype=float)
        rotation = np.eye(3, dtype=float)
        moving_index = 0
        points = [position.copy()]
        for offset, origin_rotation, axis in self.chain:
            position = position + rotation @ offset
            rotation = rotation @ origin_rotation
            if axis is not None:
                rotation = rotation @ Rotation.from_rotvec(axis * q[moving_index]).as_matrix()
                moving_index += 1
            points.append(position.copy())
        points = np.asarray(points, dtype=float)
        return points @ self.world_from_base[:3, :3].T + self.world_from_base[:3, 3]

    def link_poses_world(self, q_rad: np.ndarray) -> dict[str, np.ndarray]:
        """World poses of URDF child-link frames along the serial chain."""
        q = np.asarray(q_rad, dtype=float).reshape(6)
        pose = np.eye(4, dtype=float)
        moving_index = 0
        result: dict[str, np.ndarray] = {}
        for child, (offset, origin_rotation, axis) in zip(self.chain_child_links, self.chain):
            joint_origin = np.eye(4, dtype=float)
            joint_origin[:3, :3] = origin_rotation
            joint_origin[:3, 3] = offset
            pose = pose @ joint_origin
            if axis is not None:
                joint_rotation = np.eye(4, dtype=float)
                joint_rotation[:3, :3] = Rotation.from_rotvec(
                    axis * q[moving_index]
                ).as_matrix()
                pose = pose @ joint_rotation
                moving_index += 1
            result[child] = self.world_from_base @ pose
        return result

    def shovel_pose_world(self, q_rad: np.ndarray) -> np.ndarray:
        return self.world_from_base @ self.flange_pose_base(q_rad) @ self.flange_from_shovel

    def tool_pose_world(
        self, q_rad: np.ndarray, tool_reference_local_m=(0.0, 0.0, 0.0)
    ) -> np.ndarray:
        pose = self.shovel_pose_world(q_rad).copy()
        pose[:3, 3] += pose[:3, :3] @ np.asarray(tool_reference_local_m, dtype=float).reshape(3)
        return pose

    def position_and_normal(self, q_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pose = self.shovel_pose_world(q_rad)
        normal = pose[:3, :3] @ self.shovel_normal_local
        return pose[:3, 3].copy(), normal / np.linalg.norm(normal)

    def position_jacobian(self, q_rad: np.ndarray, step: float = 2.0e-5) -> np.ndarray:
        """Analytic world-frame translational Jacobian of the shovel origin."""
        return self.spatial_jacobian(q_rad)[:3]

    def spatial_jacobian(
        self,
        q_rad: np.ndarray,
        tool_reference_local_m=(0.0, 0.0, 0.0),
        step: float = 2.0e-5,
    ) -> np.ndarray:
        """Analytic world-frame Jacobian [linear; angular] at a shovel-local point."""
        q = np.asarray(q_rad, dtype=float).reshape(6)
        pose = self.world_from_base.copy()
        joint_origins = []
        joint_axes = []
        moving_index = 0
        for offset, origin_rotation, axis in self.chain:
            joint_origin = np.eye(4, dtype=float)
            joint_origin[:3, :3] = origin_rotation
            joint_origin[:3, 3] = offset
            pose = pose @ joint_origin
            if axis is None:
                continue
            joint_origins.append(pose[:3, 3].copy())
            joint_axes.append(pose[:3, :3] @ axis)
            joint_rotation = np.eye(4, dtype=float)
            joint_rotation[:3, :3] = Rotation.from_rotvec(
                axis * q[moving_index]
            ).as_matrix()
            pose = pose @ joint_rotation
            moving_index += 1

        point = self.tool_pose_world(q, tool_reference_local_m)[:3, 3]
        jacobian = np.zeros((6, 6), dtype=float)
        for index, (origin, axis_world) in enumerate(zip(joint_origins, joint_axes)):
            jacobian[:3, index] = np.cross(axis_world, point - origin)
            jacobian[3:, index] = axis_world
        return jacobian

    def tool_twist_world(
        self,
        q_rad: np.ndarray,
        qd_rad_s: np.ndarray,
        tool_reference_local_m=(0.0, 0.0, 0.0),
    ) -> np.ndarray:
        return self.spatial_jacobian(q_rad, tool_reference_local_m) @ np.asarray(
            qd_rad_s, dtype=float
        ).reshape(6)

    def solve_full_pose(
        self,
        seed_q_rad: np.ndarray,
        target_position_world_m: np.ndarray,
        target_rotation_world: np.ndarray,
        *,
        tool_reference_local_m=(0.0, 0.0, 0.0),
        max_evaluations: int = 180,
        position_tolerance_m: float = 0.005,
        orientation_tolerance_deg: float = 2.0,
    ) -> FullIKResult:
        """Bounded IK for a complete world-frame pose at a shovel-local tool point."""
        seed = np.clip(np.asarray(seed_q_rad, dtype=float).reshape(6), self.low, self.high)
        target_position = np.asarray(target_position_world_m, dtype=float).reshape(3)
        target_rotation = np.asarray(target_rotation_world, dtype=float).reshape(3, 3)
        if not np.isfinite(target_position).all() or not np.isfinite(target_rotation).all():
            raise ValueError("Full-pose IK target must be finite")
        orthogonality = target_rotation.T @ target_rotation
        if not np.allclose(orthogonality, np.eye(3), atol=1.0e-5) or np.linalg.det(target_rotation) < 0.0:
            raise ValueError("target_rotation_world must be a proper rotation matrix")
        joint_span = np.maximum(self.high - self.low, np.deg2rad(1.0))
        position_scale = max(position_tolerance_m, 1.0e-4)
        orientation_scale = max(np.deg2rad(orientation_tolerance_deg), 1.0e-4)

        def residual(q_value: np.ndarray) -> np.ndarray:
            pose = self.tool_pose_world(q_value, tool_reference_local_m)
            orientation_error = Rotation.from_matrix(
                pose[:3, :3] @ target_rotation.T
            ).as_rotvec()
            return np.r_[
                (pose[:3, 3] - target_position) / position_scale,
                orientation_error / orientation_scale,
                0.001 * (q_value - seed) / joint_span,
            ]

        solution = least_squares(
            residual,
            seed,
            bounds=(self.low, self.high),
            max_nfev=max_evaluations,
            ftol=1.0e-10,
            xtol=1.0e-10,
            gtol=1.0e-10,
        )
        pose = self.tool_pose_world(solution.x, tool_reference_local_m)
        position_error = float(np.linalg.norm(pose[:3, 3] - target_position))
        orientation_error = float(
            np.degrees(
                Rotation.from_matrix(pose[:3, :3] @ target_rotation.T).magnitude()
            )
        )
        reachable = bool(
            np.isfinite(solution.x).all()
            and np.all(solution.x >= self.low - 1.0e-10)
            and np.all(solution.x <= self.high + 1.0e-10)
            and position_error <= position_tolerance_m
            and orientation_error <= orientation_tolerance_deg
        )
        return FullIKResult(
            q_rad=np.asarray(solution.x, dtype=float),
            position_world_m=pose[:3, 3].copy(),
            rotation_world=pose[:3, :3].copy(),
            position_error_m=position_error,
            orientation_error_deg=orientation_error,
            optimizer_success=bool(solution.success),
            reachable=reachable,
            cost=float(solution.cost),
            evaluations=int(solution.nfev),
        )

    def solve(
        self,
        seed_q_rad: np.ndarray,
        target_position_world_m: np.ndarray,
        target_normal_world: np.ndarray,
        *,
        max_evaluations: int = 100,
        position_tolerance_m: float = 0.005,
        normal_tolerance_deg: float = 2.0,
    ) -> IKResult:
        """Bounded least-squares IK that leaves roll about the normal unconstrained."""
        seed = np.clip(np.asarray(seed_q_rad, dtype=float).reshape(6), self.low, self.high)
        target_position = np.asarray(target_position_world_m, dtype=float).reshape(3)
        target_normal = np.asarray(target_normal_world, dtype=float).reshape(3)
        target_normal_norm = float(np.linalg.norm(target_normal))
        if not np.isfinite(target_position).all() or not np.isfinite(target_normal_norm):
            raise ValueError("IK target must be finite")
        if target_normal_norm < 1.0e-12:
            raise ValueError("IK target normal must be non-zero")
        target_normal /= target_normal_norm
        joint_span = np.maximum(self.high - self.low, np.deg2rad(1.0))
        orientation_scale = np.sin(np.deg2rad(2.0))

        def residual(q_rad: np.ndarray) -> np.ndarray:
            position, normal = self.position_and_normal(q_rad)
            return np.r_[
                (position - target_position) / 0.005,
                (normal - target_normal) / orientation_scale,
                0.002 * (q_rad - seed) / joint_span,
            ]

        solution = least_squares(
            residual,
            seed,
            bounds=(self.low, self.high),
            max_nfev=max_evaluations,
            ftol=1.0e-9,
            xtol=1.0e-9,
            gtol=1.0e-9,
        )
        position, normal = self.position_and_normal(solution.x)
        position_error = float(np.linalg.norm(position - target_position))
        normal_error = float(
            np.degrees(np.arccos(np.clip(np.dot(normal, target_normal), -1.0, 1.0)))
        )
        # Least-squares may hit its evaluation cap after already reaching a
        # valid task-space solution. Feasibility is therefore based on the
        # achieved residual and joint bounds; optimizer_success is reported
        # separately for diagnostics.
        reachable = bool(
            np.isfinite(solution.x).all()
            and np.all(solution.x >= self.low - 1.0e-10)
            and np.all(solution.x <= self.high + 1.0e-10)
            and position_error <= position_tolerance_m
            and normal_error <= normal_tolerance_deg
        )
        return IKResult(
            q_rad=np.asarray(solution.x, dtype=float),
            position_world_m=position,
            normal_world=normal,
            position_error_m=position_error,
            normal_error_deg=normal_error,
            optimizer_success=bool(solution.success),
            reachable=reachable,
            cost=float(solution.cost),
            evaluations=int(solution.nfev),
        )
