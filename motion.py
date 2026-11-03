"""Task-space action curves, continuous joint trajectories, and explicit audits."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
from scipy.interpolate import PPoly, make_interp_spline
from scipy.special import betainc
from scipy.spatial.transform import Rotation

import bootstrap  # Register the bundled runtime before importing its modules.
from design import fixed_start, target_dict, validate, write_json
from emcp_phase1_planning import EMCPPhase1Context
from trajectory_planning import quintic_boundary_motion
from motion_join import concatenate


def shape_parameters(shape):
    return {"early": (3, 6), "balanced": (4, 4), "late": (6, 3)}[shape]


def progress(t, start, end, shape):
    a, b = shape_parameters(shape)
    return betainc(a, b, np.clip((np.asarray(t) - start) / (end - start), 0, 1))


def carried_time(t, duration, delay, shape):
    """Integral of a unit speed plateau followed by a smooth monotone brake."""
    a, b = shape_parameters(shape)
    t = np.clip(np.asarray(t), 0, duration)
    start = delay * duration
    window = duration - start
    x = np.clip((t - start) / window, 0, 1)
    integral = x * (1 - betainc(a, b, x)) + a / (a + b) * betainc(a + 1, b, x)
    return np.minimum(t, start) + window * integral


def desired_pose(spec, pose0, twist0, times):
    a = spec["action"]
    T = a["duration_s"]
    travel = carried_time(times, T, a["brake_start_fraction"], a["brake_shape"])
    # Initial world angular velocity also decays smoothly; flip is an additional local rotation.
    angular_travel = carried_time(times, T, 0., "balanced")
    angle = np.deg2rad(a["flip_angle_deg"]) * progress(
        times, T * a["flip_start_fraction"], T * a["flip_end_fraction"], a["flip_shape"])
    axis = np.eye(3)[("tool_x", "tool_y", "tool_z").index(a["flip_axis"])]
    poses = []
    for dt, dw, theta in zip(np.atleast_1d(travel), np.atleast_1d(angular_travel), np.atleast_1d(angle)):
        p = np.eye(4)
        p[:3, 3] = pose0[:3, 3] + twist0[:3] * dt
        p[:3, :3] = Rotation.from_rotvec(twist0[3:] * dw).as_matrix() @ pose0[:3, :3] @ Rotation.from_rotvec(axis * theta).as_matrix()
        poses.append(p)
    return np.asarray(poses)


def spline_motion(times, q, qd0, qdd0, qd1=None, qdd1=None):
    qd1 = np.zeros(6) if qd1 is None else qd1
    qdd1 = np.zeros(6) if qdd1 is None else qdd1
    spline = make_interp_spline(times, q, k=5,
                               bc_type=([(1, qd0), (2, qdd0)], [(1, qd1), (2, qdd1)]))
    knots = np.unique(spline.t)
    pieces = [quintic_boundary_motion(spline(x), spline(x, 1), spline(x, 2),
                                     spline(y), spline(y, 1), spline(y, 2), y - x)
              for x, y in zip(knots[:-1], knots[1:])]
    return PPoly(np.concatenate([p.c for p in pieces], axis=1), knots)


def polynomial_limits(motion, low, high, velocity, acceleration, jerk):
    """Exact per-piece extrema, plus left/right limits at every join."""
    extrema = []
    for order in range(4):
        poly = motion.derivative(order)
        mins, maxs = np.full(6, np.inf), np.full(6, -np.inf)
        for i, duration in enumerate(np.diff(poly.x)):
            for j in range(6):
                c = poly.c[:, i, j]
                roots = np.roots(np.polyder(c)) if len(c) > 1 else []
                ts = [0., duration] + [float(r.real) for r in roots if abs(r.imag) < 1e-8 and 0 < r.real < duration]
                values = np.polyval(c, ts)
                mins[j] = min(mins[j], min(values))
                maxs[j] = max(maxs[j], max(values))
        extrema.append((mins, maxs))
    jumps = []
    for order in range(3):
        poly = motion.derivative(order)
        jump = 0.
        for i in range(1, len(poly.x) - 1):
            left = np.array([np.polyval(poly.c[:, i - 1, j], poly.x[i] - poly.x[i - 1]) for j in range(6)])
            jump = max(jump, float(np.max(np.abs(left - poly.c[-1, i]))))
        jumps.append(jump)
    ratios = [float(np.max(np.maximum(abs(mn), abs(mx)) / lim))
              for (mn, mx), lim in zip(extrema[1:], (velocity, acceleration, jerk))]
    return {"position_valid": bool(np.all(extrema[0][0] >= low - 1e-8) and np.all(extrema[0][1] <= high + 1e-8)),
            "velocity_ratio": ratios[0], "acceleration_ratio": ratios[1], "jerk_ratio": ratios[2],
            "join_jumps_q_qd_qdd": jumps,
            "valid": bool(np.all(extrema[0][0] >= low - 1e-8) and np.all(extrema[0][1] <= high + 1e-8)
                          and max(ratios) <= 1.001 and max(jumps) <= 1e-6)}


def exact_audit(context, motion):
    return polynomial_limits(motion, context.kinematics.low, context.kinematics.high,
                             context.joint_velocity_limits * context.settings["joint_velocity_hard_fraction"],
                             context.joint_acceleration_limits * context.settings["joint_acceleration_hard_fraction"],
                             context.joint_jerk_limits)


def audit(context, motion):
    end = context.kinematics.tool_pose_world(motion(motion.x[-1]), context.tool_reference)
    check = copy.copy(context)
    check.target_position, check.target_rotation = end[:3, 3], end[:3, :3]
    twist = context.kinematics.tool_twist_world(motion(motion.x[-1]), motion(motion.x[-1], 1), context.tool_reference)
    check.target_linear_velocity, check.target_angular_velocity = twist[:3], twist[3:]
    report = check.audit_motion(motion)
    report["polynomial"] = exact_audit(context, motion)
    report["valid"] = bool(report["valid"] and report["polynomial"]["valid"])
    report["collision_scope"] = "sampled link capsules / shovel geometry; not continuous mesh certification"
    return report


def failure_message(report):
    reasons = []
    p = report["polynomial"]
    for name in ("velocity", "acceleration", "jerk"):
        if p[name + "_ratio"] > 1.001:
            reasons.append(f"joint {name} {p[name + '_ratio']:.2f}x limit")
    if not p["position_valid"]:
        reasons.append("joint position limit")
    if max(p["join_jumps_q_qd_qdd"]) > 1e-6:
        reasons.append("discontinuous phase join")
    if report["tcp_max_ratio"] > 1.005:
        reasons.append(f"TCP speed/acceleration {report['tcp_max_ratio']:.2f}x limit")
    if report["minimum_bucket_clearance_m"] < .02:
        reasons.append("pot clearance below 20 mm")
    if report["minimum_self_clearance_m"] < 0:
        reasons.append("self collision")
    if not reasons:
        reasons.append("workspace, detour, joint travel or endpoint audit")
    return "; ".join(reasons)


def build_release(spec, output, status=lambda message: None):
    validate(spec)
    status("Solving the locked release pose and velocity")
    context = EMCPPhase1Context(target=target_dict(spec), endpoint_trial_count=9, endpoint_max_solutions=1)
    q0, qd0 = context.q_goal.copy(), context.qd_goal.copy()
    pose0 = context.kinematics.tool_pose_world(q0, context.tool_reference)
    twist0 = context.kinematics.tool_twist_world(q0, qd0, context.tool_reference)
    J = context.kinematics.spatial_jacobian(q0, context.tool_reference)
    eps = 1e-4
    Jdot = (context.kinematics.spatial_jacobian(q0 + eps * qd0, context.tool_reference)
            - context.kinematics.spatial_jacobian(q0 - eps * qd0, context.tool_reference)) / (2 * eps)
    qdd0 = np.linalg.lstsq(J, -Jdot @ qd0, rcond=None)[0]
    T = spec["action"]["duration_s"]
    # Include all action event boundaries rather than sampling only a uniform clock.
    a = spec["action"]
    times = np.unique(np.r_[np.linspace(0, T, 33), T * np.array([
        a["brake_start_fraction"], a["flip_start_fraction"], a["flip_end_fraction"]])])
    # Avoid almost-duplicate knot times entered through floating point sliders.
    times = times[np.r_[True, np.diff(times) > 1e-6]]
    poses = desired_pose(spec, pose0, twist0, times)
    qs = [q0]
    for i, p in enumerate(poses[1:], 1):
        result = context.kinematics.solve_full_pose(qs[-1], p[:3, 3], p[:3, :3],
                    tool_reference_local_m=context.tool_reference, max_evaluations=140,
                    position_tolerance_m=1e-6, orientation_tolerance_deg=1e-4)
        if not result.reachable or np.max(abs(result.q_rad - qs[-1])) > .4:
            raise ValueError(f"Action IK failed or changed branch at {times[i]:.3f}s. Reduce the flip/translation or change the release pose.")
        qs.append(result.q_rad)
    motion = spline_motion(times, np.asarray(qs), qd0, qdd0)
    status("Checking full action limits, C2 joins and sampled collisions")
    report = audit(context, motion)
    sample = sample_motion(context, motion)
    desired = desired_pose(spec, pose0, twist0, sample["times_s"])
    position_error = np.linalg.norm(sample["tool_position"] - desired[:, :3, 3], axis=1)
    orientation_error = np.rad2deg(Rotation.from_matrix(sample["tool_rotation"] @ np.swapaxes(desired[:, :3, :3], 1, 2)).magnitude())
    report["action_position_error_max_m"] = float(max(position_error))
    report["action_orientation_error_max_deg"] = float(max(orientation_error))
    report["valid"] &= bool(max(position_error) <= .005 and max(orientation_error) <= 2.)
    report["actual_release"] = {"q": q0, "qd": qd0, "qdd": qdd0, "p": pose0[:3, 3], "R": pose0[:3, :3], "twist": twist0}
    report["follow_distance_m"] = float(np.linalg.norm(sample["tool_position"][-1] - pose0[:3, 3]))
    report["fixed_start"] = fixed_start()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "release_audit.json", report)
    np.savez_compressed(output / "release_samples.npz", **sample)
    if not report["valid"]:
        reason = failure_message(report)
        if max(position_error) > .005 or max(orientation_error) > 2:
            reason += "; task-space action tracking tolerance"
        raise ValueError("Action rejected: " + reason + ". Try a longer duration, smaller flip, earlier braking, or a different release state.")
    save_motion(output / "release.npz", motion, context, release_time=0.)
    return context, motion, report


def sample_motion(context, motion):
    times = np.linspace(0, motion.x[-1], max(121, int(np.ceil(motion.x[-1] * 120)) + 1))
    q, qd = motion(times), motion(times, 1)
    poses = np.asarray([context.kinematics.tool_pose_world(x, context.tool_reference) for x in q])
    twists = np.asarray([context.kinematics.tool_twist_world(x, v, context.tool_reference) for x, v in zip(q, qd)])
    return {"times_s": times, "q": q, "qd": qd, "qdd": motion(times, 2), "jerk": motion(times, 3),
            "tool_position": poses[:, :3, 3], "tool_rotation": poses[:, :3, :3], "twist": twists}


def save_motion(path, motion, context, release_time):
    times = np.linspace(0, motion.x[-1], max(2, int(motion.x[-1] * 60) + 1))
    q = motion(times)
    flange = [context.kinematics.flange_pose_base(x) for x in q]
    poses = np.asarray([np.r_[p[:3, 3], Rotation.from_matrix(p[:3, :3]).as_quat()] for p in flange])
    start = context.kinematics.tool_pose_world(motion(0), context.tool_reference)
    twist = context.kinematics.tool_twist_world(motion(0), motion(0, 1), context.tool_reference)
    np.savez_compressed(path, coefficients=motion.c, t=motion.x, q=q, poses=poses,
                        release_start_s=release_time, release_state_s=release_time,
                        release_action_start_s=release_time,
                        start_tool_position_world_m=start[:3, 3],
                        start_linear_velocity_world_m_s=twist[:3], start_angular_velocity_world_rad_s=twist[3:])


def load_motion(path):
    with np.load(path, allow_pickle=False) as d:
        return PPoly(d["coefficients"], d["t"])
