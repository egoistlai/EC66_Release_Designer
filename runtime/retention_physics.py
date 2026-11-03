"""Cheap shovel-frame particle-retention screen for Type08 planning.

The model is deliberately simpler than Newton.  It evaluates the apparent
acceleration at the settled initial particle locations and integrates only
outward motion towards the open lip.  Its purpose is to reject and rank
trajectory candidates before particle simulation, not to certify retention.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh


HERE = Path(__file__).resolve().parent
NEWTON_SCENE = HERE


def load_material_model(tool_reference_local_m: np.ndarray) -> dict:
    scene = json.loads((HERE / "scene_config.json").read_text(encoding="utf-8"))
    particles_config = json.loads(
        (NEWTON_SCENE / "scene_config_particles.json").read_text(encoding="utf-8")
    )
    snapshot = json.loads(
        (NEWTON_SCENE / particles_config["particle_positions_file"]).read_text(encoding="utf-8")
    )
    particle_local = np.asarray(snapshot["positions_m"], dtype=float)
    raw = trimesh.load(HERE / scene["shovel_mesh"], force="mesh")
    vertices = (
        np.asarray(raw.vertices, dtype=float) - np.asarray(scene["shovel_origin_stl"], dtype=float)
    ) * float(scene["shovel_scale"])
    # STL axes: old Z -> shovel X (lip), old X -> shovel Y, old Y -> shovel Z (opening).
    vertices = vertices[:, [2, 0, 1]]
    lip_x = float(np.max(vertices[:, 0]))
    lip_band = vertices[vertices[:, 0] >= lip_x - 0.015]
    lip_center_local = np.array(
        [lip_x, float(np.median(lip_band[:, 1])), float(np.median(lip_band[:, 2]))]
    )
    return {
        "particle_positions_shovel_m": particle_local,
        "particle_offsets_from_tool_m": particle_local
        - np.asarray(tool_reference_local_m, dtype=float).reshape(3),
        "particle_radius_m": 0.5 * float(snapshot["particle_diameter_m"]),
        "static_friction": float(particles_config["friction"]),
        # A lower dynamic coefficient makes the screen conservative once slip starts.
        "dynamic_friction": 0.75 * float(particles_config["friction"]),
        "lip_x_shovel_m": lip_x,
        "lip_center_offset_from_tool_m": lip_center_local
        - np.asarray(tool_reference_local_m, dtype=float).reshape(3),
        "bucket_center_world_m": np.asarray(scene["bucket_xyz"], dtype=float),
        "bucket_radius_m": float(scene["bucket_radius"]),
    }


def retention_physics_audit(
    context,
    motion,
    *,
    sample_step_s: float = 0.01,
    pot_edge_margin_m: float = 0.05,
) -> dict:
    """Estimate pre-Release slip and expose interpretable risk features."""
    duration = float(motion.x[-1] - motion.x[0])
    count = max(121, int(np.ceil(duration / sample_step_s)) + 1)
    times = np.linspace(float(motion.x[0]), float(motion.x[-1]), count)
    dt = float(times[1] - times[0])
    q = np.asarray(motion(times), dtype=float)
    qd = np.asarray(motion(times, 1), dtype=float)
    poses = np.asarray(
        [context.kinematics.tool_pose_world(value, context.tool_reference) for value in q]
    )
    twists = np.asarray(
        [
            context.kinematics.tool_twist_world(value, velocity, context.tool_reference)
            for value, velocity in zip(q, qd)
        ]
    )
    linear_acceleration_world = np.gradient(twists[:, :3], times, axis=0, edge_order=2)
    angular_acceleration_world = np.gradient(twists[:, 3:], times, axis=0, edge_order=2)
    rotations = poses[:, :3, :3]
    gravity_world = np.array([0.0, 0.0, -9.81])
    model = load_material_model(context.tool_reference)
    offsets = model["particle_offsets_from_tool_m"]
    particle_x = model["particle_positions_shovel_m"][:, 0].copy()
    particle_radius = float(model["particle_radius_m"])
    lip_limit_x = float(model["lip_x_shovel_m"] - particle_radius)
    relative_speed = np.zeros(len(offsets), dtype=float)
    slip_distance = np.zeros(len(offsets), dtype=float)
    escaped = np.zeros(len(offsets), dtype=bool)
    escape_time = np.full(len(offsets), np.nan, dtype=float)
    escape_pot_margin = np.full(len(offsets), np.nan, dtype=float)
    retained_curve = np.empty(count, dtype=int)
    minimum_margin = np.inf
    negative_margin_auc = 0.0
    maximum_outward_acceleration = 0.0
    maximum_lift_acceleration = 0.0
    lip_pot_margin = np.empty(count, dtype=float)
    static_mu = float(model["static_friction"])
    dynamic_mu = float(model["dynamic_friction"])

    for index, (rotation, pose, omega_world, alpha_world, acceleration_world) in enumerate(
        zip(
            rotations,
            poses,
            twists[:, 3:],
            angular_acceleration_world,
            linear_acceleration_world,
        )
    ):
        world_to_shovel = rotation.T
        omega = world_to_shovel @ omega_world
        alpha = world_to_shovel @ alpha_world
        base = world_to_shovel @ (gravity_world - acceleration_world)
        effective = (
            base[None, :]
            - np.cross(alpha[None, :], offsets)
            - np.cross(omega[None, :], np.cross(omega[None, :], offsets))
        )
        normal_load = np.maximum(-effective[:, 2], 0.0)
        static_margin = static_mu * normal_load - np.maximum(effective[:, 0], 0.0)
        front = particle_x >= np.quantile(particle_x, 0.75)
        minimum_margin = min(minimum_margin, float(np.min(static_margin[front])))
        negative_margin_auc += float(np.mean(np.maximum(-static_margin[front], 0.0))) * dt
        maximum_outward_acceleration = max(
            maximum_outward_acceleration, float(np.max(np.maximum(effective[:, 0], 0.0)))
        )
        maximum_lift_acceleration = max(
            maximum_lift_acceleration, float(np.max(np.maximum(effective[:, 2], 0.0)))
        )

        moving = relative_speed > 1.0e-6
        outward_acceleration = np.where(
            moving,
            effective[:, 0] - dynamic_mu * normal_load,
            np.maximum(effective[:, 0] - static_mu * normal_load, 0.0),
        )
        next_speed = relative_speed + outward_acceleration * dt
        stopped = moving & (next_speed < 0.0)
        next_speed[stopped] = 0.0
        next_speed = np.maximum(next_speed, 0.0)
        slip_distance += 0.5 * (relative_speed + next_speed) * dt
        relative_speed = next_speed

        lip_world = pose[:3, 3] + rotation @ model["lip_center_offset_from_tool_m"]
        center = model["bucket_center_world_m"]
        lip_pot_margin[index] = (
            float(model["bucket_radius_m"])
            - pot_edge_margin_m
            - float(np.linalg.norm(lip_world[:2] - center[:2]))
        )
        newly_escaped = (~escaped) & (particle_x + slip_distance >= lip_limit_x)
        escape_time[newly_escaped] = float(times[index])
        escape_pot_margin[newly_escaped] = lip_pot_margin[index]
        escaped |= newly_escaped
        retained_curve[index] = int(np.count_nonzero(~escaped))

    first_escape = np.flatnonzero(escaped)
    first_escape_time = float(np.nanmin(escape_time)) if len(first_escape) else None
    unsafe_before_capture = escaped & (escape_pot_margin < 0.0)
    return {
        "model": "shovel_frame_effective_acceleration_plus_1d_lip_slip_v1",
        "particle_count": int(len(offsets)),
        "predicted_retained_count_at_release": int(retained_curve[-1]),
        "predicted_retained_fraction_at_release": float(retained_curve[-1] / len(offsets)),
        "predicted_first_escape_time_s": first_escape_time,
        "predicted_escaped_before_pot_capture_count": int(np.count_nonzero(unsafe_before_capture)),
        "predicted_escaped_before_pot_capture_fraction": float(
            np.count_nonzero(unsafe_before_capture) / len(offsets)
        ),
        "minimum_front_quartile_friction_margin_m_s2": float(minimum_margin),
        "negative_front_friction_margin_auc_m_s": float(negative_margin_auc),
        "maximum_outward_effective_acceleration_m_s2": float(maximum_outward_acceleration),
        "maximum_opening_lift_acceleration_m_s2": float(maximum_lift_acceleration),
        "maximum_predicted_slip_distance_m": float(np.max(slip_distance)),
        "p95_predicted_slip_distance_m": float(np.quantile(slip_distance, 0.95)),
        "lip_pot_margin_at_release_m": float(lip_pot_margin[-1]),
        "maximum_lip_pot_margin_m": float(np.max(lip_pot_margin)),
        "static_friction": static_mu,
        "dynamic_friction": dynamic_mu,
        "sample_step_s": dt,
    }
