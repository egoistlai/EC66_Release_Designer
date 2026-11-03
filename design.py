"""Versioned release intent, immutable experiment snapshots and local storage."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
from scipy.spatial.transform import Rotation

from bootstrap import ROOT, RUNTIME


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    from trajectory_planning import json_safe
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def fingerprint(spec):
    value = {k: v for k, v in spec.items() if k != "name"}
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def fixed_start():
    """The migrated BASE-FRAME FLANGE start, not an old world-frame tool pose."""
    cfg = read_json(RUNTIME / "scene_config.json")
    return {"q_rad": np.deg2rad(cfg["initial_q_deg"]).tolist(),
            "flange_pose_base": cfg["initial_flange_tcp_pose_base"],
            "source": str(RUNTIME / "scene_config.json")}


def default_design():
    from kinematics import EC66Kinematics, transform, mount_transform
    cfg = read_json(RUNTIME / "scene_config.json")
    kin = EC66Kinematics(RUNTIME / cfg["urdf"],
                        transform(cfg["world_from_base_xyz"], np.deg2rad(cfg["world_from_base_rpy_deg"])),
                        mount_transform(cfg["mount_rotations_deg"], cfg["mount_xyz"]), cfg["shovel_normal_local"])
    pose = kin.tool_pose_world(np.asarray(fixed_start()["q_rad"]), np.array([.3, 0, .08]))
    return {"schema": "ec66_release_design", "version": 1, "name": "My release",
            "target": {"p_world_m": pose[:3, 3].tolist(),
                       "rpy_xyz_deg": Rotation.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True).tolist(),
                       "v_world_m_s": [0.15, -0.20, 0.10], "omega_world_rad_s": [0., 0., 0.]},
            "action": {"duration_s": 1.0, "brake_start_fraction": 0.1,
                       "brake_shape": "balanced", "flip_axis": "tool_y", "flip_angle_deg": -10.,
                       "flip_start_fraction": .15, "flip_end_fraction": .85, "flip_shape": "balanced"},
            "simulation": {"substeps": 4, "settle_seconds": 4., "load_mode": "ideal", "seed": 7,
                           "velocity_noise_m_s": .02},
            "approach": {"allow_pullback": True, "screen_candidates": 3, "minimum_retention": .95}}


def validate(spec):
    if spec.get("schema") != "ec66_release_design" or spec.get("version") != 1:
        raise ValueError("Unsupported release-design schema/version")
    for key in ("p_world_m", "rpy_xyz_deg", "v_world_m_s", "omega_world_rad_s"):
        a = np.asarray(spec["target"][key], dtype=float)
        if a.shape != (3,) or not np.isfinite(a).all():
            raise ValueError(f"{key} must contain three finite numbers")
    a, sim, pre = spec["action"], spec["simulation"], spec["approach"]
    if not .15 <= a["duration_s"] <= 4.:
        raise ValueError("Action duration must be 0.15–4 seconds")
    if not 0 <= a["brake_start_fraction"] <= .8:
        raise ValueError("Braking must start within the first 80% of the action")
    if not 0 <= a["flip_start_fraction"] < a["flip_end_fraction"] <= 1:
        raise ValueError("Flip start must be earlier than flip end")
    if a["flip_end_fraction"] - a["flip_start_fraction"] < .1:
        raise ValueError("Flip window must span at least 10% of the action")
    if not np.isfinite(a["flip_angle_deg"]) or abs(a["flip_angle_deg"]) > 120:
        raise ValueError("Flip angle must be within +/-120 degrees")
    if a["flip_axis"] not in ("tool_x", "tool_y", "tool_z"):
        raise ValueError("Choose a tool-local flip axis")
    if any(a[k] not in ("early", "balanced", "late") for k in ("brake_shape", "flip_shape")):
        raise ValueError("Unknown timing shape")
    if sim["substeps"] not in (4, 8, 12) or not .5 <= sim["settle_seconds"] <= 15:
        raise ValueError("Use 4/8/12 substeps and 0.5–15 seconds settling")
    if sim["load_mode"] not in ("ideal", "velocity_jitter"):
        raise ValueError("Unknown load mode")
    if not 0 <= sim["velocity_noise_m_s"] <= .2:
        raise ValueError("Relative-velocity noise must be 0–0.2 m/s")
    if not 1 <= pre["screen_candidates"] <= 6 or not 0 <= pre["minimum_retention"] <= 1:
        raise ValueError("Invalid approach screening settings")
    # Also rejects NaN in scalar settings, unsupported values, and malformed seeds.
    json.dumps(spec, allow_nan=False)
    if not isinstance(sim["seed"], int) or sim["seed"] < 0:
        raise ValueError("Random seed must be a nonnegative integer")
    if not isinstance(pre["screen_candidates"], int):
        raise ValueError("Screen candidate count must be an integer")


def target_dict(spec):
    t = spec["target"]
    return {"p_world_m": np.asarray(t["p_world_m"]),
            "R_world_from_tool": Rotation.from_euler("xyz", t["rpy_xyz_deg"], degrees=True).as_matrix(),
            "v_world_m_s": np.asarray(t["v_world_m_s"]),
            "omega_world_rad_s": np.asarray(t["omega_world_rad_s"])}


def save_design(spec):
    validate(spec)
    slug = re.sub(r"[^\w-]+", "_", spec["name"], flags=re.UNICODE).strip("_")[:60] or "release"
    path = ROOT / "schemes" / f"{slug}_{datetime.now(ZoneInfo('Asia/Shanghai')):%Y%m%d_%H%M%S}_{uuid4().hex[:6]}.json"
    write_json(path, spec)
    return path


def create_run(spec, task):
    validate(spec)
    directory = ROOT / "outputs" / f"{datetime.now(ZoneInfo('Asia/Shanghai')):%Y%m%d_%H%M%S}_{task}_{uuid4().hex[:6]}"
    write_json(directory / "design.json", spec)
    write_json(directory / "status.json", {"state": "queued", "message": "Queued", "task": task,
                                          "design_hash": fingerprint(spec)})
    return directory
