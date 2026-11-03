"""Distribution overlay math, independent of training datasets."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
CFG = json.loads((Path(__file__).resolve().parent / 'scene_config_particles.json').read_text())

def pot_mask(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    center = np.asarray(CFG["bucket_xyz"], dtype=float)
    rotation = Rotation.from_euler("xyz", CFG.get("bucket_rpy_deg", [0, 0, 0]), degrees=True).as_matrix()
    local = (points - center) @ rotation
    radial = np.linalg.norm(local[:, :2], axis=1)
    return (
        (radial < float(CFG["bucket_radius"]) - float(CFG["bucket_wall"]))
        & (local[:, 2] >= float(CFG["bucket_wall"]) - float(CFG["voxel_size"]))
        & (local[:, 2] < float(CFG["bucket_height"]) - 0.01)
    )

def gaussian_w2(mean, covariance, goal):
    cholesky = np.array([[np.exp(goal[2]), 0.0], [goal[3], np.exp(goal[4])]])
    target_covariance = cholesky @ cholesky.T
    covariance = np.asarray(covariance, dtype=float)
    squared = (
        np.sum((np.asarray(mean) - goal[:2]) ** 2)
        + np.trace(covariance + target_covariance)
        - 2 * np.sqrt(max(0.0, np.trace(covariance @ target_covariance) + 2 * np.sqrt(max(0.0, np.linalg.det(covariance) * np.linalg.det(target_covariance)))))
    )
    return float(np.sqrt(max(0.0, squared)))
