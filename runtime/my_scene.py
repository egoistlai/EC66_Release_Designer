# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""EC66 + chanzi.STL + open bucket + MPM material.

The arm follows prescribed joint coordinates (one-way kinematic coupling).
This is a scene/trajectory prototype, not a torque-controlled robot simulation.
See SCENE_README.md for configuration, assumptions and commands.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import PchipInterpolator
import trimesh
import warp as wp
import newton
import newton.examples
from newton.solvers import SolverImplicitMPM

HERE = Path(__file__).resolve().parent


def configure_auto_num_frames(parser):
    """Set Newton's viewer frame limit from the configured trajectory duration."""
    if any(arg == '--num-frames' or arg.startswith('--num-frames=') for arg in sys.argv[1:]):
        return
    known, _unknown = parser.parse_known_args()
    config_path = Path(known.config).resolve()
    cfg = json.loads(config_path.read_text())
    pre = float(cfg.get('settle_seconds', 0.0))
    post = float(cfg.get('post_settle_seconds', 1.0))
    if min(pre, post) < 0:
        raise ValueError('settle_seconds and post_settle_seconds must be nonnegative')
    if known.static:
        motion = 0.0
    elif cfg.get('joint_trajectory_enabled', False):
        trajectory = np.loadtxt(config_path.parent / cfg['joint_trajectory_file'], comments='#', dtype=float)
        if trajectory.ndim == 1:
            trajectory = trajectory[None, :]
        motion = float(trajectory[-1, 0]) * float(cfg.get('trajectory_time_scale', 1.0))
    else:
        motion = float(cfg.get('throw_seconds', 0.0))
    duration = pre + motion + post
    frames = max(1, int(math.ceil(duration * float(cfg['fps']))))
    parser.set_defaults(num_frames=frames)
    print(f'Auto frame count: {frames} frames = {duration:.3f} s '
          f'(pre={pre:.3f}, motion={motion:.3f}, post={post:.3f})')


def transform(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
    t[:3, 3] = xyz
    return t


def warp_transform(t):
    return wp.transform(wp.vec3(*t[:3, 3]), wp.quat(*Rotation.from_matrix(t[:3, :3]).as_quat()))


def flange_fk(root, angles):
    """URDF chain FK, also used before a Newton model has been finalized."""
    poses = {'world': np.eye(4)}
    pending = list(root.findall('joint'))
    while pending:
        progressed = False
        for joint in pending[:]:
            parent = joint.find('parent').get('link')
            if parent not in poses:
                continue
            o = joint.find('origin')
            xyz = np.fromstring(o.get('xyz', '0 0 0'), sep=' ') if o is not None else np.zeros(3)
            rpy = np.fromstring(o.get('rpy', '0 0 0'), sep=' ') if o is not None else np.zeros(3)
            t = transform(xyz, rpy)
            if joint.get('type') == 'revolute':
                axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
                rotation = np.eye(4)
                rotation[:3, :3] = Rotation.from_rotvec(axis * angles[joint.get('name')]).as_matrix()
                t = t @ rotation
            poses[joint.find('child').get('link')] = poses[parent] @ t
            pending.remove(joint)
            progressed = True
        if not progressed:
            raise ValueError('URDF chain must connect to world')
    return poses['flan']


def fill_shovel(mesh, cfg):
    """Sample above the upper mesh surface, without an optional rtree dependency.

    Shovel frame: +X towards lip, +Y across blade, +Z towards opening.
    The initial material top is horizontal in the shovel frame.
    """
    dx = cfg['voxel_size']
    spacing = float(cfg.get('particle_diameter_m', dx / cfg['particles_per_cell']))
    if spacing <= 0:
        raise ValueError('particle_diameter_m must be positive')
    lo, hi = np.array(cfg['fill_lo']), np.array(cfg['fill_hi'])
    tri = mesh.triangles
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac = b - a, c - a
    det = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
    valid = np.abs(det) > 1.e-12
    a, ab, ac, det = a[valid], ab[valid], ac[valid], det[valid]
    points = []
    for x in np.arange(lo[0] + spacing / 2, hi[0], spacing):
        for y in np.arange(lo[1] + spacing / 2, hi[1], spacing):
            px, py = x - a[:, 0], y - a[:, 1]
            u = (px * ac[:, 1] - py * ac[:, 0]) / det
            v = (ab[:, 0] * py - ab[:, 1] * px) / det
            inside = (u >= 0) & (v >= 0) & (u + v <= 1)
            if not inside.any():
                continue
            surface = np.max((a[:, 2] + u * ab[:, 2] + v * ac[:, 2])[inside])
            clearance = max(dx, cfg.get('shovel_collision_margin', 0.0) + spacing)
            bottom = max(lo[2], surface + clearance)
            for z in np.arange(bottom + spacing / 2, hi[2], spacing):
                points.append((x, y, z))
    if not points:
        raise ValueError('No fill particles: check shovel scale/origin and fill_lo/fill_hi')
    return np.asarray(points, dtype=np.float32), spacing


class Example:
    def __init__(self, viewer, args):
        self.viewer, self.args = viewer, args
        config_path = Path(args.config).resolve()
        self.cfg = c = json.loads(config_path.read_text())
        self.sim_time = 0.0
        self.frame_index = 0
        self.frame_dt = 1.0 / c['fps']
        self.dt = self.frame_dt / c['substeps']
        self.trajectory_enabled = bool(c.get('joint_trajectory_enabled', False))
        self.trajectory_times = self.trajectory_q = self.trajectory_spline = self.trajectory_spline_d = None
        if self.trajectory_enabled:
            trajectory_path = config_path.parent / c['joint_trajectory_file']
            trajectory = np.loadtxt(trajectory_path, comments='#', dtype=float)
            if trajectory.ndim == 1:
                trajectory = trajectory[None, :]
            if trajectory.ndim != 2 or trajectory.shape[1] < 7 or len(trajectory) < 2:
                raise ValueError('Joint trajectory must contain at least two rows: time_sec q1..q6')
            time_scale = float(c.get('trajectory_time_scale', 1.0))
            if time_scale <= 0:
                raise ValueError('trajectory_time_scale must be positive')
            self.trajectory_times = trajectory[:, 0] * time_scale
            self.trajectory_q = trajectory[:, 1:7]
            if abs(self.trajectory_times[0]) > 1e-9 or not np.all(np.diff(self.trajectory_times) > 0):
                raise ValueError('Joint trajectory time must start at zero and be strictly increasing')
            self.trajectory_spline = PchipInterpolator(self.trajectory_times, self.trajectory_q, axis=0)
            self.trajectory_spline_d = self.trajectory_spline.derivative()
            self.q0 = self.trajectory_q[0].copy()
            self.q1 = self.trajectory_q[-1].copy()
        else:
            self.q0 = np.deg2rad(c['initial_q_deg'])
            self.q1 = np.deg2rad(c['throw_q_deg'])
        if len(self.q0) != 6 or len(self.q1) != 6:
            raise ValueError('Expected six joint angles')
        if min(c['voxel_size'], c['particles_per_cell'], c['fps'], c['substeps'], c['throw_seconds']) <= 0:
            raise ValueError('Resolution, timing and particle parameters must be positive')
        if c.get('shovel_collision_margin', 0.0) < 0:
            raise ValueError('shovel_collision_margin must be nonnegative')
        urdf_path = config_path.parent / c['urdf']
        root = ET.parse(urdf_path).getroot()
        self.world_from_base = transform(
            c['world_from_base_xyz'], np.deg2rad(c['world_from_base_rpy_deg'])
        )
        for mesh in root.findall('.//mesh'):
            path = (urdf_path.parent / mesh.get('filename')).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            mesh.set('filename', str(path))
        # The supplied flan link contains zero-size box placeholders.
        # Remove only these placeholders from the in-memory import copy.
        for link in root.findall('link'):
            for kind in ('visual', 'collision'):
                for elem in list(link.findall(kind)):
                    box = elem.find('geometry/box')
                    if box is not None and np.any(np.fromstring(box.get('size'), sep=' ') <= 0):
                        link.remove(elem)
        all_commanded_q = self.trajectory_q if self.trajectory_enabled else np.vstack((self.q0, self.q1))
        self.joint_velocity_limits = np.empty(6, dtype=float)
        for i in range(6):
            limit = root.find(f"joint[@name='joint{i+1}']/limit")
            if limit is None:
                raise ValueError(f'joint{i+1} has no URDF limit')
            if np.any(all_commanded_q[:, i] < float(limit.get('lower'))) or np.any(
                all_commanded_q[:, i] > float(limit.get('upper'))):
                raise ValueError(f'joint{i+1} target outside URDF limits')
            self.joint_velocity_limits[i] = float(limit.get('velocity'))
        configured_velocity_deg = c.get('joint_velocity_limits_deg_s')
        self.joint_velocity_limit_source = 'URDF'
        if configured_velocity_deg is not None:
            configured_velocity_deg = np.asarray(configured_velocity_deg, dtype=float)
            if configured_velocity_deg.shape != (6,) or np.any(configured_velocity_deg <= 0):
                raise ValueError('joint_velocity_limits_deg_s must contain six positive values')
            self.joint_velocity_limits = np.deg2rad(configured_velocity_deg)
            self.joint_velocity_limit_source = 'teach pendant configuration'
        self.trajectory_max_velocity = self.trajectory_max_acceleration = None
        self.trajectory_endpoint_velocity = None
        self.trajectory_tcp_peaks = None
        if self.trajectory_enabled:
            velocity_scale = float(c.get('joint_velocity_limit_scale', 1.0))
            if not 0.0 < velocity_scale <= 1.0:
                raise ValueError('joint_velocity_limit_scale must be in (0, 1]')
            check_times = np.linspace(
                self.trajectory_times[0], self.trajectory_times[-1],
                max(1000, len(self.trajectory_times) * 50),
            )
            sampled_qd = self.trajectory_spline_d(check_times)
            sampled_qdd = self.trajectory_spline.derivative(2)(check_times)
            self.trajectory_max_velocity = np.max(np.abs(sampled_qd), axis=0)
            self.trajectory_max_acceleration = np.max(np.abs(sampled_qdd), axis=0)
            self.trajectory_endpoint_velocity = np.vstack((sampled_qd[0], sampled_qd[-1]))
            allowed_velocity = self.joint_velocity_limits * velocity_scale
            exceeded = np.flatnonzero(self.trajectory_max_velocity > allowed_velocity + 1e-9)
            if len(exceeded):
                details = ', '.join(
                    f'joint{i+1}: {self.trajectory_max_velocity[i]:.3f} > {allowed_velocity[i]:.3f} rad/s'
                    for i in exceeded
                )
                raise ValueError(f'Joint trajectory exceeds configured velocity limits: {details}')
            acceleration_limits = c.get('joint_acceleration_limits_deg_s2')
            if acceleration_limits is not None:
                acceleration_limits = np.deg2rad(np.asarray(acceleration_limits, dtype=float))
            else:
                acceleration_limits = c.get('joint_acceleration_limits_rad_s2')
            if acceleration_limits is not None:
                acceleration_limits = np.asarray(acceleration_limits, dtype=float)
                if acceleration_limits.shape != (6,) or np.any(acceleration_limits <= 0):
                    raise ValueError('joint acceleration limits must contain six positive values')
                exceeded = np.flatnonzero(self.trajectory_max_acceleration > acceleration_limits + 1e-9)
                if len(exceeded):
                    details = ', '.join(
                        f'joint{i+1}: {self.trajectory_max_acceleration[i]:.3f} > '
                        f'{acceleration_limits[i]:.3f} rad/s^2' for i in exceeded
                    )
                    raise ValueError(f'Joint trajectory exceeds configured acceleration limits: {details}')

            flange_samples = [
                flange_fk(root, dict(zip([f'joint{i+1}' for i in range(6)], q_sample)))
                for q_sample in self.trajectory_spline(check_times)
            ]
            positions = np.asarray([sample[:3, 3] for sample in flange_samples])
            rotations = np.asarray([sample[:3, :3] for sample in flange_samples])
            sample_dt = check_times[1] - check_times[0]
            linear_velocity = np.diff(positions, axis=0) / sample_dt
            linear_acceleration = np.diff(linear_velocity, axis=0) / sample_dt
            angular_velocity = np.asarray([
                Rotation.from_matrix(rotations[i + 1] @ rotations[i].T).as_rotvec() / sample_dt
                for i in range(len(rotations) - 1)
            ])
            angular_acceleration = np.diff(angular_velocity, axis=0) / sample_dt
            self.trajectory_tcp_peaks = np.asarray([
                np.max(np.linalg.norm(linear_velocity, axis=1)),
                np.max(np.linalg.norm(linear_acceleration, axis=1)),
                np.max(np.linalg.norm(angular_velocity, axis=1)),
                np.max(np.linalg.norm(angular_acceleration, axis=1)),
            ])
            tcp_limits = np.asarray([
                c.get('tcp_linear_velocity_limit_m_s', np.inf),
                c.get('tcp_linear_acceleration_limit_m_s2', np.inf),
                np.deg2rad(c.get('tcp_angular_velocity_limit_deg_s', np.inf)),
                np.deg2rad(c.get('tcp_angular_acceleration_limit_deg_s2', np.inf)),
            ], dtype=float)
            if np.any(tcp_limits <= 0):
                raise ValueError('TCP velocity and acceleration limits must be positive')
            labels = ('linear velocity m/s', 'linear acceleration m/s^2',
                      'angular velocity rad/s', 'angular acceleration rad/s^2')
            exceeded = np.flatnonzero(self.trajectory_tcp_peaks > tcp_limits + 1e-9)
            if len(exceeded):
                details = ', '.join(
                    f'{labels[i]}: {self.trajectory_tcp_peaks[i]:.3f} > {tcp_limits[i]:.3f}'
                    for i in exceeded
                )
                raise ValueError(f'TCP trajectory exceeds configured limits: {details}')
        self.trajectory_max_position_error = self.trajectory_max_orientation_error = 0.0
        if self.trajectory_enabled:
            pose_samples = np.loadtxt(config_path.parent / c['pose_trajectory_file'], comments='#', dtype=float)
            if pose_samples.ndim == 1:
                pose_samples = pose_samples[None, :]
            if len(pose_samples) != len(self.trajectory_q) or pose_samples.shape[1] < 7:
                raise ValueError('Pose and joint trajectory sample counts/formats do not match')
            for row, q_sample in zip(pose_samples, self.trajectory_q):
                target = np.eye(4)
                target[:3, 3] = row[:3]
                target[:3, :3] = Rotation.from_quat(row[3:7] / np.linalg.norm(row[3:7])).as_matrix()
                actual = flange_fk(root, dict(zip([f'joint{i+1}' for i in range(6)], q_sample)))
                self.trajectory_max_position_error = max(
                    self.trajectory_max_position_error, np.linalg.norm(actual[:3, 3] - target[:3, 3]))
                self.trajectory_max_orientation_error = max(
                    self.trajectory_max_orientation_error,
                    np.linalg.norm(Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).as_rotvec()))
            if self.trajectory_max_position_error > 1e-3 or self.trajectory_max_orientation_error > np.deg2rad(0.2):
                raise ValueError('Cached joint trajectory no longer matches the configured pose trajectory')
        initial_flange_base = flange_fk(root, dict(zip([f'joint{i+1}' for i in range(6)], self.q0)))
        initial_flange = self.world_from_base @ initial_flange_base
        if not self.trajectory_enabled:
            requested_ee_base = transform(
                c['initial_ee_xyz_base'], np.deg2rad(c['initial_ee_rpy_deg_base'])
            )
            ee_position_error = np.linalg.norm(initial_flange_base[:3, 3] - requested_ee_base[:3, 3])
            ee_angle_error = np.linalg.norm(
                Rotation.from_matrix(requested_ee_base[:3, :3] @ initial_flange_base[:3, :3].T).as_rotvec()
            )
            if ee_position_error > 5e-4 or ee_angle_error > np.deg2rad(0.1):
                raise ValueError(
                    'initial_q_deg does not match initial_ee pose: '
                    f'position error={ee_position_error:.6g} m, angle error={np.rad2deg(ee_angle_error):.6g} deg'
                )
        # The shovel origin is coincident with the flange origin. Apply the
        # configured rotations in order about the FIXED flange axes. Positive
        # angles follow the right-hand rule when looking along the axis toward
        # the origin: +90 is counter-clockwise, -90 is clockwise.
        if not np.allclose(c['mount_xyz'], 0.0):
            raise ValueError('The shovel and flange origins must stay coincident: mount_xyz must be zero')
        mount = transform(c['mount_xyz'])
        axes = {'x': np.array([1.0, 0.0, 0.0]),
                'y': np.array([0.0, 1.0, 0.0]),
                'z': np.array([0.0, 0.0, 1.0])}
        for item in c['mount_rotations_deg']:
            axis = item['axis'].lower()
            if axis not in axes:
                raise ValueError(f'Invalid mount rotation axis: {axis!r}')
            fixed_axis_rotation = Rotation.from_rotvec(
                axes[axis] * np.deg2rad(float(item['angle_deg']))
            ).as_matrix()
            mount[:3, :3] = fixed_axis_rotation @ mount[:3, :3]
        self.initial_shovel = initial_flange @ mount
        self.mount = mount
        raw = trimesh.load(config_path.parent / c['shovel_mesh'], force='mesh')
        vertices = (np.asarray(raw.vertices) - c['shovel_origin_stl']) * c['shovel_scale']
        # STL axes: old Z -> forward, old X -> lateral, old Y -> opening.
        vertices = vertices[:, [2, 0, 1]]
        self.shovel_mesh = trimesh.Trimesh(vertices=vertices, faces=raw.faces, process=False)
        if not self.shovel_mesh.is_watertight:
            raise ValueError('Shovel STL must be watertight for signed-distance collision')
        builder = newton.ModelBuilder()
        builder.add_urdf(ET.tostring(root, encoding='unicode'), xform=warp_transform(self.world_from_base), floating=False,
                         collapse_fixed_joints=False, enable_self_collisions=False)
        # Prescribed FK: robot links are visual only; MPM sees shovel, bucket, ground.
        for i in range(builder.shape_count):
            builder.shape_flags[i] &= ~int(newton.ShapeFlags.COLLIDE_PARTICLES | newton.ShapeFlags.COLLIDE_SHAPES)
        self.flange_id = next(i for i, name in enumerate(builder.body_label) if name.split('/')[-1] == 'flan')
        cfg = newton.ModelBuilder.ShapeConfig(density=0, mu=c['friction'], has_shape_collision=False)
        builder.add_shape_mesh(self.flange_id, xform=warp_transform(mount),
                               mesh=newton.Mesh(vertices=vertices.astype(np.float32), indices=np.asarray(raw.faces, np.int32).ravel()),
                               cfg=cfg, label='shovel', color=(0.65, 0.67, 0.70))
        self._build_bucket(builder, cfg)
        ground_z = c['bucket_xyz'][2]
        self.ground_mesh = trimesh.creation.box(extents=(10, 10, 0.5))
        self.ground_mesh.apply_translation((0, 0, ground_z - 0.25))
        builder.add_shape_box(-1, xform=warp_transform(transform((0, 0, ground_z - 0.25))),
                              hx=5.0, hy=5.0, hz=0.25, cfg=cfg, label='ground',
                              color=(0.3, 0.3, 0.32))
        self.joint_q_ids, self.joint_dof_ids = [], []
        for i, angle in enumerate(self.q0):
            joint = next(j for j, name in enumerate(builder.joint_label) if name.split('/')[-1] == f'joint{i+1}')
            qi, di = builder.joint_q_start[joint], builder.joint_qd_start[joint]
            builder.joint_q[qi] = float(angle)
            self.joint_q_ids.append(qi)
            self.joint_dof_ids.append(di)
        self.model = builder.finalize()
        print(f'Rigid/MPM simulation device: {self.model.device}')
        self.state = self.model.state()
        self.q_host = self.model.joint_q.numpy()
        self.qd_host = self.model.joint_qd.numpy()
        self._set_arm(0.0)
        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.sand_model = self.sand_state = self.mpm_solver = None
        if c.get('particle_positions_file'):
            snapshot_path = config_path.parent / c['particle_positions_file']
            snapshot = json.loads(snapshot_path.read_text())
            if snapshot.get('coordinate_frame') != 'shovel_local':
                raise ValueError('Particle position configuration must use shovel_local coordinates')
            local = np.asarray(snapshot['positions_m'], dtype=np.float32)
            count = int(c['fixed_particle_count'])
            if local.shape != (count, 3) or not np.isfinite(local).all():
                raise ValueError('Saved particle positions do not match configured count')
            if not np.isclose(snapshot['particle_diameter_m'], c['particle_diameter_m']):
                raise ValueError('Saved particle configuration has a different diameter')
            spacing = float(c['particle_diameter_m'])
        else:
            local, spacing = fill_shovel(self.shovel_mesh, c)
            count = int(c.get('fixed_particle_count', len(local)))
            if count <= 0 or count > len(local):
                raise ValueError(f'fixed_particle_count must be in 1..{len(local)}')
            local = local[np.linspace(0, len(local)-1, count, dtype=int)]
        self.fill_local = local.copy()
        positions = local @ self.initial_shovel[:3, :3].T + self.initial_shovel[:3, 3]
        self.initial_particles = positions.astype(np.float32)
        print(f'Shovel extent (m): {self.shovel_mesh.extents}; initial flange in world: {initial_flange[:3, 3]}')
        print('Flange-to-shovel origin: coincident (xyz=0)')
        print('Flange-to-shovel fixed-axis rotations:', c['mount_rotations_deg'])
        print('Flange-to-shovel rotation matrix:\n', mount[:3, :3])
        if self.trajectory_enabled:
            print(f'Joint trajectory: {c["joint_trajectory_file"]}, '
                  f'{len(self.trajectory_times)} points, {self.trajectory_times[-1]:.6f} s')
            print(f'IK verification: max position={self.trajectory_max_position_error * 1000:.6g} mm, '
                  f'max orientation={np.rad2deg(self.trajectory_max_orientation_error):.6g} deg')
            print('Peak joint velocity (deg/s):',
                  np.array2string(np.rad2deg(self.trajectory_max_velocity), precision=2))
            print(f'Joint velocity limits from {self.joint_velocity_limit_source} (deg/s):',
                  np.array2string(np.rad2deg(self.joint_velocity_limits), precision=2))
            print('Peak joint acceleration (deg/s^2):',
                  np.array2string(np.rad2deg(self.trajectory_max_acceleration), precision=1))
            print('Peak TCP linear velocity/acceleration:',
                  f'{self.trajectory_tcp_peaks[0]:.3f} m/s, {self.trajectory_tcp_peaks[1]:.3f} m/s^2')
            print('Peak TCP angular velocity/acceleration:',
                  f'{np.rad2deg(self.trajectory_tcp_peaks[2]):.2f} deg/s, '
                  f'{np.rad2deg(self.trajectory_tcp_peaks[3]):.2f} deg/s^2')
            if np.max(np.abs(self.trajectory_endpoint_velocity)) > 1e-6:
                print('WARNING: trajectory endpoint velocity is nonzero; the pre/post hold introduces an '
                      'instantaneous velocity change. Use endpoint blending for acceleration-continuous motion.')
        print(f'Particles: {len(positions)}; bucket bottom center: {c["bucket_xyz"]}; coupling: prescribed FK')
        if not args.scene_only:
            sand = newton.ModelBuilder()
            SolverImplicitMPM.register_custom_attributes(sand)
            for p in positions:
                sand.add_particle(pos=wp.vec3(*p), vel=wp.vec3(0.0), mass=c['density'] * (math.pi / 6 * spacing**3 if 'particle_diameter_m' in c else spacing**3),
                                  radius=spacing / 2, custom_attributes={'mpm:friction': c['friction'],
                                  **{f'mpm:{key}': value for key,value in c.get('material_parameters', {}).items()}})
            self.sand_model = sand.finalize()
            self.sand_state = self.sand_model.state()
            self.sand_state.body_q = wp.clone(self.state.body_q)
            self.sand_state.body_qd = wp.clone(self.state.body_qd)
            self.sand_state.body_f = wp.zeros_like(self.state.body_f)
            options = SolverImplicitMPM.Config()
            options.voxel_size = c['voxel_size']
            options.grid_type = 'sparse'
            options.max_iterations = int(c.get('mpm_max_iterations', 50))
            if options.max_iterations <= 0:
                raise ValueError('mpm_max_iterations must be positive')
            options.strain_basis = 'P0'
            options.critical_fraction = 0.0
            options.collider_velocity_mode = 'backward'
            self.mpm_solver = SolverImplicitMPM(self.sand_model, config=options)
            # Keep the two closed static surfaces independent. Combining
            # intersecting solids into one triangle soup breaks signed distance.
            self.static_collider_meshes = [
                wp.Mesh(points=wp.array(m.vertices.astype(np.float32), dtype=wp.vec3),
                        indices=wp.array(m.faces.astype(np.int32).ravel(), dtype=int))
                for m in (self.ground_mesh, self.bucket_mesh)
            ]
            self.mpm_solver.setup_collider(model=self.model, body_mass=wp.zeros_like(self.model.body_mass),
                                           body_q=self.state.body_q,
                                           collider_meshes=[*self.static_collider_meshes, None],
                                           collider_body_ids=[None, None, self.flange_id],
                                           collider_margins=[0.0, 0.0, c.get('shovel_collision_margin', 0.0)],
                                           collider_friction=[c['friction']] * 3)
        self.preview_points = wp.array(self.initial_particles, dtype=wp.vec3)
        # The OpenGL upload path calls colors.numpy(), even though log_points
        # advertises tuple colors. Allocate one RGB value per material point.
        self.particle_colors = wp.full(len(self.initial_particles),
                                       value=wp.vec3(0.75, 0.6, 0.3), dtype=wp.vec3,
                                       device=self.model.device)

    def _build_bucket(self, builder, cfg):
        c = self.cfg
        self.world_from_bucket = transform(c['bucket_xyz'], np.deg2rad(c['bucket_rpy_deg']))
        r, h, wall, n = c['bucket_radius'], c['bucket_height'], c['bucket_wall'], c['bucket_segments']
        if min(r, h, wall) <= 0 or n < 12:
            raise ValueError('Invalid bucket dimensions/segments')
        if wall >= h:
            raise ValueError('Bucket bottom must be thinner than bucket height')
        # Revolve the solid wall/floor cross-section. The cavity stays open;
        # the material surface is watertight, with no intersecting panel seams.
        self.bucket_mesh = trimesh.creation.revolve(
            [[0, 0], [r + wall, 0], [r + wall, h], [r, h], [r, wall], [0, wall], [0, 0]],
            sections=n,
        )
        self.bucket_mesh.apply_transform(self.world_from_bucket)
        if not self.bucket_mesh.is_watertight:
            raise ValueError('Bucket mesh is not watertight')
        builder.add_shape_mesh(-1, mesh=newton.Mesh(vertices=self.bucket_mesh.vertices.astype(np.float32),
                               indices=self.bucket_mesh.faces.astype(np.int32).ravel()),
                               cfg=cfg, color=(0.25, 0.5, 0.65), label='bucket')

    def _set_arm(self, t):
        c = self.cfg
        if self.args.static:
            q_command = self.q0
            qd_command = np.zeros(6)
        elif self.trajectory_enabled:
            trajectory_t = t - c['settle_seconds']
            if trajectory_t <= 0:
                q_command = self.q0
                qd_command = np.zeros(6)
            elif trajectory_t >= self.trajectory_times[-1]:
                q_command = self.q1
                qd_command = np.zeros(6)
            else:
                q_command = self.trajectory_spline(trajectory_t)
                qd_command = self.trajectory_spline_d(trajectory_t)
        else:
            u = np.clip((t - c['settle_seconds']) / c['throw_seconds'], 0, 1)
            blend = 10*u**3 - 15*u**4 + 6*u**5
            derivative = (30*u**2 - 60*u**3 + 30*u**4) / c['throw_seconds']
            q_command = self.q0 + blend * (self.q1 - self.q0)
            qd_command = derivative * (self.q1 - self.q0)
        self.q_host[self.joint_q_ids] = q_command
        self.qd_host.fill(0)
        self.qd_host[self.joint_dof_ids] = qd_command
        self.state.joint_q.assign(self.q_host)
        self.state.joint_qd.assign(self.qd_host)
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)

    def step(self):
        for _ in range(self.cfg['substeps']):
            self.sim_time += self.dt
            self._set_arm(self.sim_time)
            if self.mpm_solver is not None:
                self.sand_state.body_q.assign(self.state.body_q)
                self.sand_state.body_qd.assign(self.state.body_qd)
                self.mpm_solver.step(self.sand_state, self.sand_state, contacts=None, control=None, dt=self.dt)
                # Grid contact alone can miss isolated material points at thin
                # surfaces. Newton's geometric projection also corrects velocity.
                self.mpm_solver.project_outside(self.sand_state, self.sand_state, dt=self.dt,
                                                gap=4 * self.cfg['voxel_size'])
        self.frame_index += 1
        progress_every = int(self.cfg.get('progress_every_frames', 30))
        if progress_every > 0 and self.frame_index % progress_every == 0:
            total = getattr(self.args, 'num_frames', None)
            suffix = f'/{total} ({100.0 * self.frame_index / total:.1f}%)' if total else ''
            print(f'[progress] frame {self.frame_index}{suffix}, simulated time {self.sim_time:.3f} s', flush=True)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        if self.sand_state is not None:
            self.viewer.log_points('/sand', points=self.sand_state.particle_q,
                                   radii=self.sand_model.particle_radius, colors=self.particle_colors)
        elif self.args.static:
            self.viewer.log_points('/initial_fill_preview', points=self.preview_points,
                                   radii=float(self.cfg.get('particle_diameter_m', self.cfg['voxel_size'] / self.cfg['particles_per_cell'])) / 2,
                                   colors=self.particle_colors)
        self.viewer.end_frame()

    def export_results(self):
        if self.sand_state is None:
            return
        out = Path(self.args.results).resolve()
        out.mkdir(parents=True, exist_ok=True)
        p, v, m = self.sand_state.particle_q.numpy(), self.sand_state.particle_qd.numpy(), self.sand_model.particle_mass.numpy()
        c = self.cfg
        r = c['bucket_radius']
        bucket_from_world = np.linalg.inv(self.world_from_bucket)
        p_bucket = p @ bucket_from_world[:3, :3].T + bucket_from_world[:3, 3]
        inside = (np.sum(p_bucket[:, :2]**2, axis=1) < r*r) & (
            p_bucket[:, 2] >= c['bucket_wall'] - c['voxel_size']
        ) & (p_bucket[:, 2] < c['bucket_height'])
        settled = inside & (np.linalg.norm(v, axis=1) < c['settled_speed'])
        mass_map, xe, ye = np.histogram2d(p_bucket[settled, 0], p_bucket[settled, 1], bins=c['distribution_bins'],
                                         range=((-r, r), (-r, r)), weights=m[settled])
        np.savez_compressed(out / 'particles_final.npz', positions=p, velocities=v, masses=m,
                            inside_bucket=inside, settled=settled, sim_time=self.sim_time)
        np.savez_compressed(out / 'bucket_mass_map.npz', mass_map=mass_map, x_edges=xe, y_edges=ye)
        summary = dict(sim_time=self.sim_time, total_mass_kg=float(m.sum()), inside_bucket_mass_kg=float(m[inside].sum()),
                       settled_bucket_mass_kg=float(m[settled].sum()), coupling='kinematic_one_way', config=c)
        (out / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(f'Results saved to {out}; settled bucket mass: {m[settled].sum():.6f} kg')

    def test_final(self):
        if not np.isfinite(self.state.body_q.numpy()).all():
            raise AssertionError('Non-finite arm transforms')
        if self.sand_state is not None and not np.isfinite(self.sand_state.particle_q.numpy()).all():
            raise AssertionError('Non-finite particle positions')
        if self.sand_state is not None:
            p = self.sand_state.particle_q.numpy()
            ground_z = self.cfg['bucket_xyz'][2]
            over_ground = (np.abs(p[:, 0]) < 5) & (np.abs(p[:, 1]) < 5)
            if np.any(p[over_ground, 2] < ground_z - 2 * self.cfg['voxel_size']):
                raise AssertionError('Particles penetrated the ground; refine the contact/time resolution')

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument('--config', default=str(HERE / 'scene_config.json'))
        parser.add_argument('--scene-only', action='store_true', help='Preview geometry without MPM')
        parser.add_argument('--static', action='store_true', help='Hold initial joint pose')
        parser.add_argument('--results', default=str(HERE / 'results'))
        return parser


if __name__ == '__main__':
    parser = Example.create_parser()
    configure_auto_num_frames(parser)
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    completed = False
    try:
        newton.examples.run(example, args)
        completed = True
    finally:
        example.export_results()
        if completed:
            print(f'Simulation completed normally at {example.sim_time:.3f} simulated seconds.')
        else:
            print(f'Simulation interrupted at {example.sim_time:.3f} simulated seconds.')
