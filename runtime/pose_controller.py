"""Interactive Newton GUI for EC66 shovel position and face-normal IK."""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import warp as wp

import newton
import newton.examples

from kinematics import (
    EC66Kinematics,
    IKResult,
    angles_from_normal,
    mount_transform,
    normal_from_angles,
    transform,
)


HERE = Path(__file__).resolve().parent


def warp_transform(matrix: np.ndarray) -> wp.transform:
    quaternion = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return wp.transform(wp.vec3(*matrix[:3, 3]), wp.quat(*quaternion))


def load_urdf_with_absolute_meshes(path: Path) -> ET.Element:
    root = ET.parse(path).getroot()
    for mesh in root.findall(".//mesh"):
        mesh_path = (path.parent / mesh.get("filename")).resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError(mesh_path)
        mesh.set("filename", str(mesh_path))
    # The supplied flange contains zero-size placeholders that Newton rejects.
    for link in root.findall("link"):
        for kind in ("visual", "collision"):
            for element in list(link.findall(kind)):
                box = element.find("geometry/box")
                if box is not None and np.any(np.fromstring(box.get("size"), sep=" ") <= 0):
                    link.remove(element)
    return root


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        config_path = Path(args.config).resolve()
        self.config_path = config_path
        self.cfg = json.loads(config_path.read_text(encoding="utf-8"))
        cfg = self.cfg
        self.sim_time = 0.0
        self.frame_dt = 1.0 / float(cfg.get("fps", 60.0))

        urdf_path = config_path.parent / cfg["urdf"]
        self.world_from_base = transform(
            cfg["world_from_base_xyz"], np.deg2rad(cfg["world_from_base_rpy_deg"])
        )
        self.mount = mount_transform(cfg["mount_rotations_deg"], cfg.get("mount_xyz", [0, 0, 0]))
        self.kinematics = EC66Kinematics(
            urdf_path,
            self.world_from_base,
            self.mount,
            cfg.get("shovel_normal_local", [0.0, 0.0, 1.0]),
        )
        self.initial_q = np.deg2rad(np.asarray(cfg["initial_q_deg"], dtype=float))
        self.q = self.initial_q.copy()
        initial_position, initial_normal = self.kinematics.position_and_normal(self.initial_q)
        self.target_position = initial_position.copy()
        self.target_azimuth_deg, self.target_elevation_deg = angles_from_normal(initial_normal)
        self.live_solve = True
        self.show_actual_marker = True
        self.solve_count = 0
        self.last_result: IKResult | None = None

        root = load_urdf_with_absolute_meshes(urdf_path)
        builder = newton.ModelBuilder()
        builder.add_urdf(
            ET.tostring(root, encoding="unicode"),
            xform=warp_transform(self.world_from_base),
            floating=False,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
        )
        for shape_index in range(builder.shape_count):
            builder.shape_flags[shape_index] &= ~int(
                newton.ShapeFlags.COLLIDE_PARTICLES | newton.ShapeFlags.COLLIDE_SHAPES
            )

        self.flange_id = next(
            index
            for index, label in enumerate(builder.body_label)
            if label.split("/")[-1] == "flan"
        )
        shape_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0, mu=float(cfg.get("friction", 0.6)), has_shape_collision=False
        )
        raw_shovel = trimesh.load(config_path.parent / cfg["shovel_mesh"], force="mesh")
        shovel_vertices = (
            np.asarray(raw_shovel.vertices) - np.asarray(cfg["shovel_origin_stl"], dtype=float)
        ) * float(cfg["shovel_scale"])
        # Supplied STL axes: old Z -> forward, old X -> lateral, old Y -> opening.
        shovel_vertices = shovel_vertices[:, [2, 0, 1]]
        builder.add_shape_mesh(
            self.flange_id,
            xform=warp_transform(self.mount),
            mesh=newton.Mesh(
                vertices=shovel_vertices.astype(np.float32),
                indices=np.asarray(raw_shovel.faces, dtype=np.int32).ravel(),
            ),
            cfg=shape_cfg,
            label="shovel",
            color=(0.65, 0.67, 0.70),
        )
        self._build_bucket(builder, shape_cfg)

        ground_z = float(cfg["bucket_xyz"][2])
        builder.add_shape_box(
            -1,
            xform=warp_transform(transform((0.0, 0.0, ground_z - 0.25))),
            hx=5.0,
            hy=5.0,
            hz=0.25,
            cfg=shape_cfg,
            label="ground",
            color=(0.30, 0.30, 0.32),
        )

        self.joint_q_ids: list[int] = []
        self.joint_dof_ids: list[int] = []
        for index, angle in enumerate(self.initial_q):
            joint_index = next(
                joint
                for joint, label in enumerate(builder.joint_label)
                if label.split("/")[-1] == f"joint{index + 1}"
            )
            q_index = int(builder.joint_q_start[joint_index])
            dof_index = int(builder.joint_qd_start[joint_index])
            builder.joint_q[q_index] = float(angle)
            self.joint_q_ids.append(q_index)
            self.joint_dof_ids.append(dof_index)

        self.model = builder.finalize()
        self.state = self.model.state()
        self.q_host = self.model.joint_q.numpy()
        self.qd_host = self.model.joint_qd.numpy()
        self._apply_joint_pose()
        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(-2.5, 1.4, 1.5), -28.0, -49.0)
        self._solve_target(use_initial_seed=True)

    def _build_bucket(self, builder, shape_cfg) -> None:
        cfg = self.cfg
        world_from_bucket = transform(cfg["bucket_xyz"], np.deg2rad(cfg["bucket_rpy_deg"]))
        radius = float(cfg["bucket_radius"])
        height = float(cfg["bucket_height"])
        wall = float(cfg["bucket_wall"])
        segments = int(cfg["bucket_segments"])
        bucket = trimesh.creation.revolve(
            [
                [0.0, 0.0],
                [radius + wall, 0.0],
                [radius + wall, height],
                [radius, height],
                [radius, wall],
                [0.0, wall],
                [0.0, 0.0],
            ],
            sections=segments,
        )
        bucket.apply_transform(world_from_bucket)
        builder.add_shape_mesh(
            -1,
            mesh=newton.Mesh(
                vertices=np.asarray(bucket.vertices, dtype=np.float32),
                indices=np.asarray(bucket.faces, dtype=np.int32).ravel(),
            ),
            cfg=shape_cfg,
            label="bucket",
            color=(0.25, 0.50, 0.65),
        )

    @property
    def target_normal(self) -> np.ndarray:
        return normal_from_angles(self.target_azimuth_deg, self.target_elevation_deg)

    def _apply_joint_pose(self) -> None:
        self.q_host[self.joint_q_ids] = self.q
        self.qd_host.fill(0.0)
        self.state.joint_q.assign(self.q_host)
        self.state.joint_qd.assign(self.qd_host)
        newton.eval_fk(
            self.model, self.state.joint_q, self.state.joint_qd, self.state
        )

    def _solve_target(self, *, use_initial_seed: bool = False) -> None:
        seed = self.initial_q if use_initial_seed else self.q
        self.last_result = self.kinematics.solve(
            seed,
            self.target_position,
            self.target_normal,
            max_evaluations=int(self.cfg.get("ik_max_evaluations", 100)),
            position_tolerance_m=float(self.cfg.get("ik_position_tolerance_m", 0.005)),
            normal_tolerance_deg=float(self.cfg.get("ik_normal_tolerance_deg", 2.0)),
        )
        self.q = self.last_result.q_rad.copy()
        self.solve_count += 1
        self._apply_joint_pose()

    def _reset_target(self) -> None:
        self.q = self.initial_q.copy()
        self.target_position, normal = self.kinematics.position_and_normal(self.initial_q)
        self.target_azimuth_deg, self.target_elevation_deg = angles_from_normal(normal)
        self._solve_target(use_initial_seed=True)

    def _save_result(self) -> Path:
        if self.last_result is None:
            raise RuntimeError("No IK result to save")
        output = (self.config_path.parent.parent / "outputs" / "last_pose.json").resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "coordinate_frame": "Newton world",
            "target_position_world_m": self.target_position.tolist(),
            "target_normal_world": self.target_normal.tolist(),
            "target_azimuth_deg": self.target_azimuth_deg,
            "target_elevation_deg": self.target_elevation_deg,
            "reachable": self.last_result.reachable,
            "solved_q_deg": np.rad2deg(self.last_result.q_rad).tolist(),
            "actual_position_world_m": self.last_result.position_world_m.tolist(),
            "actual_normal_world": self.last_result.normal_world.tolist(),
            "position_error_mm": self.last_result.position_error_m * 1000.0,
            "normal_error_deg": self.last_result.normal_error_deg,
        }
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved pose: {output}", flush=True)
        return output

    def step(self) -> None:
        self.sim_time += self.frame_dt
        self._apply_joint_pose()

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)

        target_position = self.target_position.astype(np.float32)
        target_tip = (self.target_position + self.target_normal * 0.25).astype(np.float32)
        self.viewer.log_points(
            "/ik_target/position",
            wp.array([target_position], dtype=wp.vec3),
            radii=0.025,
            colors=wp.array([(1.0, 0.15, 0.08)], dtype=wp.vec3),
        )
        self.viewer.log_arrows(
            "/ik_target/normal",
            wp.array([target_position], dtype=wp.vec3),
            wp.array([target_tip], dtype=wp.vec3),
            (0.05, 0.90, 1.0),
            width=0.012,
        )

        if self.last_result is not None and self.show_actual_marker:
            actual_position = self.last_result.position_world_m.astype(np.float32)
            actual_tip = (
                self.last_result.position_world_m + self.last_result.normal_world * 0.20
            ).astype(np.float32)
            actual_color = (0.10, 0.95, 0.20) if self.last_result.reachable else (1.0, 0.65, 0.05)
            self.viewer.log_points(
                "/ik_solution/position",
                wp.array([actual_position], dtype=wp.vec3),
                radii=0.016,
                colors=wp.array([actual_color], dtype=wp.vec3),
            )
            self.viewer.log_arrows(
                "/ik_solution/normal",
                wp.array([actual_position], dtype=wp.vec3),
                wp.array([actual_tip], dtype=wp.vec3),
                actual_color,
                width=0.009,
            )
        else:
            self.viewer.log_points("/ik_solution/position", None)
            self.viewer.log_arrows("/ik_solution/normal", None, None, None)
        self.viewer.end_frame()

    def gui(self, ui) -> None:
        ui.text("EC66 shovel pose controller")
        ui.text("World-frame target position (m)")
        changed_x, x = ui.slider_float("X", float(self.target_position[0]), -2.0, 1.0, "%.4f")
        changed_y, y = ui.slider_float("Y", float(self.target_position[1]), -2.0, 1.0, "%.4f")
        changed_z, z = ui.slider_float("Z", float(self.target_position[2]), -0.7, 1.5, "%.4f")
        changed_position = changed_x or changed_y or changed_z
        if changed_position:
            self.target_position[:] = (x, y, z)

        ui.separator()
        ui.text("Shovel face normal (world)")
        changed_azimuth, azimuth = ui.slider_float(
            "Azimuth", float(self.target_azimuth_deg), -180.0, 180.0, "%.1f deg"
        )
        changed_elevation, elevation = ui.slider_float(
            "Elevation", float(self.target_elevation_deg), -89.9, 89.9, "%.1f deg"
        )
        if changed_azimuth:
            self.target_azimuth_deg = azimuth
        if changed_elevation:
            self.target_elevation_deg = elevation
        normal = self.target_normal
        ui.text(f"normal = [{normal[0]:+.3f}, {normal[1]:+.3f}, {normal[2]:+.3f}]")

        ui.separator()
        _changed_live, self.live_solve = ui.checkbox("Live solve", self.live_solve)
        _changed_marker, self.show_actual_marker = ui.checkbox(
            "Show achieved marker", self.show_actual_marker
        )
        target_changed = changed_position or changed_azimuth or changed_elevation
        if target_changed and self.live_solve:
            self._solve_target()
        if ui.button("Solve IK"):
            self._solve_target()
        ui.same_line()
        if ui.button("Reset"):
            self._reset_target()
        if ui.button("Save target + solution"):
            self._save_result()

        ui.separator()
        if self.last_result is None:
            ui.text("No solution")
            return
        result = self.last_result
        ui.text("REACHABLE" if result.reachable else "UNREACHABLE / BEST EFFORT")
        ui.text(f"Position error: {result.position_error_m * 1000.0:.3f} mm")
        ui.text(f"Normal error: {result.normal_error_deg:.3f} deg")
        ui.text(f"Optimizer evaluations: {result.evaluations}")
        ui.text("Solved joints (deg)")
        joints_deg = np.rad2deg(result.q_rad)
        ui.text("  ".join(f"J{i + 1} {value:+.2f}" for i, value in enumerate(joints_deg[:3])))
        ui.text("  ".join(f"J{i + 4} {value:+.2f}" for i, value in enumerate(joints_deg[3:])))
        ui.separator()
        ui.text("Red point / cyan arrow: target")
        ui.text("Green: achieved; orange: best effort")

    def test_final(self) -> None:
        if self.last_result is None or not self.last_result.reachable:
            raise AssertionError("Default target must be IK-reachable")
        if self.last_result.position_error_m > 1.0e-5:
            raise AssertionError("Default target position did not reproduce the initial pose")
        if self.last_result.normal_error_deg > 1.0e-3:
            raise AssertionError("Default target normal did not reproduce the initial pose")
        if not np.isfinite(self.state.body_q.numpy()).all():
            raise AssertionError("Newton body transforms contain non-finite values")
        expected_flange = self.world_from_base @ self.kinematics.flange_pose_base(self.q)
        actual_flange = self.state.body_q.numpy()[self.flange_id]
        if np.linalg.norm(actual_flange[:3] - expected_flange[:3, 3]) > 1.0e-6:
            raise AssertionError("Newton FK and controller FK disagree on flange position")
        actual_rotation = Rotation.from_quat(actual_flange[3:]).as_matrix()
        rotation_error = Rotation.from_matrix(
            expected_flange[:3, :3] @ actual_rotation.T
        ).magnitude()
        if rotation_error > 1.0e-5:
            raise AssertionError("Newton FK and controller FK disagree on flange orientation")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--config", default=str(HERE / "scene_config.json"))
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, arguments = newton.examples.init(parser)
    newton.examples.run(Example(viewer, arguments), arguments)
