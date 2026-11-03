"""A bounded multi-seed fixed-start search, followed by actual Newton ranking."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from design import fixed_start, write_json
from motion import spline_motion, exact_audit, audit, failure_message, save_motion
from trajectory_planning import quintic_boundary_motion
from generate_type01_pitch_dump import concatenate
from retention_physics import retention_physics_audit
from physics import simulate


def plan_approach(spec, context, release, output, status=lambda m: None):
    output = Path(output)
    q0 = np.asarray(fixed_start()["q_rad"])
    qr, vr, ar = [release(0, d) for d in range(3)]
    zero = np.zeros(6)
    candidates, attempts = [], []

    def consider(prefix, label):
        status(f"Checking approach {len(attempts) + 1}: {label}; {len(candidates)} valid so far")
        exact = exact_audit(context, prefix)
        if not exact["valid"]:
            attempts.append({"label": label, "rejected": "polynomial_limits", "audit": exact})
            return
        full = concatenate(prefix, release)
        check = audit(context, full)
        if not check["valid"]:
            attempts.append({"label": label, "rejected": failure_message(check)})
            return
        physics = retention_physics_audit(context, prefix, sample_step_s=.04)
        score = (physics["negative_front_friction_margin_auc_m_s"]
                 + 2 * physics["predicted_escaped_before_pot_capture_fraction"] + .002 * prefix.x[-1])
        candidates.append({"label": label, "prefix": prefix, "full": full, "audit": check, "score": float(score)})
        attempts.append({"label": label, "valid": True, "surrogate_score": score})

    status("Searching direct continuous approaches from the original fixed flange start")
    for duration in (1.5, 2.5, 3.5, 5., 7.):
        consider(quintic_boundary_motion(q0, zero, zero, qr, vr, ar, duration), f"direct_{duration:g}s")
    pose = context.kinematics.tool_pose_world(qr, context.tool_reference)
    start_pose = context.kinematics.tool_pose_world(q0, context.tool_reference)
    # Low targets inside the pot need a different geometric route, not just slower timing.
    # Preserve the carried attitude while crossing the rim, then enter the locked release state.
    for lift in (.15, .30, .45):
        status(f"Searching rim-clearance route with {lift:.2f}m lift")
        height = max(start_pose[2, 3], pose[2, 3]) + lift
        if height > context.settings['workspace_max_world_m'][2]:
            continue
        lifted_start, lifted_goal = start_pose[:3, 3].copy(), pose[:3, 3].copy()
        lifted_start[2] = lifted_goal[2] = height
        via = [q0]
        for position in (lifted_start, lifted_goal):
            ik = context.kinematics.solve_full_pose(via[-1], position, start_pose[:3, :3],
                     tool_reference_local_m=context.tool_reference, max_evaluations=180,
                     position_tolerance_m=.002, orientation_tolerance_deg=.5)
            if not ik.reachable or not context.collision.state_valid(ik.q_rad):
                break
            via.append(ik.q_rad)
        if len(via) != 3:
            attempts.append({'label': f'clearance_{lift}', 'rejected': 'lifted waypoint IK/collision'})
            continue
        for total in (4., 6., 8.):
            prefix = spline_motion(total * np.array([0., .25, .65, 1.]),
                                   np.asarray([*via, qr]), zero, zero, vr, ar)
            consider(prefix, f'clearance_{lift}_{total}s')
    if spec["approach"]["allow_pullback"]:
        velocity = context.kinematics.tool_twist_world(qr, vr, context.tool_reference)[:3]
        direction = velocity / max(np.linalg.norm(velocity), 1e-9)
        if np.linalg.norm(velocity) < .01:
            direction = pose[:3, 2]
        seed_number = 0
        for distance in (.15, .30, .45):
            for orientation_name, orientation in (("start_attitude", start_pose[:3, :3]), ("release_attitude", pose[:3, :3])):
                p = pose[:3, 3] - distance * direction
                ik = context.kinematics.solve_full_pose(q0, p, orientation, tool_reference_local_m=context.tool_reference,
                                                       max_evaluations=180, position_tolerance_m=.002, orientation_tolerance_deg=.5)
                seed_number += 1
                status(f"Searching pullback seed {seed_number}/6; {len(candidates)} mechanically valid candidates")
                if not ik.reachable or not context.collision.state_valid(ik.q_rad):
                    attempts.append({"label": f"pullback_{distance}_{orientation_name}", "rejected": "preparation IK/collision"})
                    continue
                for total in (3., 4.5, 6.):
                    for fraction in (.40, .65):
                        # Via-point velocities are free: one spline, no artificial stop at each waypoint.
                        prefix = spline_motion(np.array([0., fraction * total, total]),
                                               np.asarray([q0, ik.q_rad, qr]), zero, zero, vr, ar)
                        consider(prefix, f"pullback_{distance}_{orientation_name}_{total}_{fraction}")
    candidates.sort(key=lambda c: c["score"])
    write_json(output / "approach_search.json", {"attempts": attempts, "mechanically_valid": len(candidates),
              "scope": "bounded direct/rim-clearance/pullback seed search, not exhaustive reachability", "fixed_start": fixed_start()})
    if not candidates:
        raise ValueError("No mechanically valid approach found in this bounded search. The release action remains valid; try a nearby release state or lower speed.")
    results = []
    for i, candidate in enumerate(candidates[:spec["approach"]["screen_candidates"]]):
        folder = output / f"approach_{i:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        release_time = float(candidate["prefix"].x[-1])
        path = folder / "full.npz"
        save_motion(path, candidate["full"], context, release_time)
        write_json(folder / "audit.json", candidate["audit"])
        status(f"Screening approach {i+1}: {candidate['label']}")
        result = simulate(path, folder, spec, context, ideal=False, release_time=release_time, status=status)
        result.update(directory=str(folder.resolve()), label=candidate["label"], surrogate_score=candidate["score"])
        result["retention_pass"] = result["minimum_pre_release_retention"] >= spec["approach"]["minimum_retention"]
        results.append(result)
    # Prioritize the measured minimum retention; relative slip breaks close ties.
    results.sort(key=lambda r: (-r["minimum_pre_release_retention"],
                               r["release_relative_speed_rms_m_s"] if r["release_relative_speed_rms_m_s"] is not None else float("inf")))
    selected = results[0]
    write_json(output / "approach_results.json", {"candidates": results, "selected": selected,
               "acceptance": "retention_pass is a single-load Newton observation, not a robust certificate"})
    return selected
