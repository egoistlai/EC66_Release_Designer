"""Versioned, inverse-planning-friendly particle distribution descriptors.

The geometric/settling conventions intentionally follow
EC66_Newton_2cm_Near_Constrained.  The primary grid is lower dimensional than
the reference 60 x 60 audit map so it can be learned from a realistically
sized Newton dataset.  Grid mass is divided by *all* particle mass; particles
that are moving, outside the pot, or still in the shovel therefore cannot be
hidden by conditional normalization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import norm


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "particle_distribution_config.json"
SCHEMA_VERSION = 1


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


@dataclass(frozen=True)
class DistributionConfig:
    bucket_center_world_m: np.ndarray
    bucket_rpy_xyz_deg: np.ndarray
    robot_base_world_xy_m: np.ndarray
    bucket_radius_m: float
    bucket_height_m: float
    bucket_wall_m: float
    bucket_segments: int
    voxel_size_m: float
    usable_radius_m: float
    settled_speed_m_s: float
    primary_grid_bins: int
    reference_grid_bins: int
    region_grid_shape: tuple[int, int]
    stability_window_s: float
    stable_centroid_drift_m: float
    stable_speed_p95_m_s: float
    quantiles: tuple[float, ...]
    comparison_weights: dict[str, float]
    coordinate_frame: str = "bucket_local"
    format: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "bucket_center_world_m", np.asarray(self.bucket_center_world_m, dtype=float).reshape(3))
        object.__setattr__(self, "bucket_rpy_xyz_deg", np.asarray(self.bucket_rpy_xyz_deg, dtype=float).reshape(3))
        object.__setattr__(self, "robot_base_world_xy_m", np.asarray(self.robot_base_world_xy_m, dtype=float).reshape(2))
        object.__setattr__(self, "region_grid_shape", tuple(int(v) for v in self.region_grid_shape))
        object.__setattr__(self, "quantiles", tuple(float(v) for v in self.quantiles))
        if self.coordinate_frame != "bucket_local":
            raise ValueError("Only the bucket_local distribution frame is supported")
        if self.primary_grid_bins < 2 or self.reference_grid_bins < 2:
            raise ValueError("Distribution grids need at least two bins")
        if any(v < 1 for v in self.region_grid_shape):
            raise ValueError("region_grid_shape must be positive")
        if self.primary_grid_bins % self.region_grid_shape[0] or self.primary_grid_bins % self.region_grid_shape[1]:
            raise ValueError("primary_grid_bins must be divisible by both region-grid dimensions")
        if not all(0.0 < q < 1.0 for q in self.quantiles):
            raise ValueError("quantiles must lie strictly inside (0, 1)")
        required_losses = {
            "grid_total_variation", "marginal_wasserstein", "region_total_variation",
            "centroid", "covariance", "axis_quantiles", "category_fractions",
        }
        if set(self.comparison_weights) != required_losses:
            raise ValueError(f"comparison_weights must contain exactly {sorted(required_losses)}")
        if any(not np.isfinite(value) or value < 0.0 for value in self.comparison_weights.values()):
            raise ValueError("comparison weights must be finite and nonnegative")

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG) -> "DistributionConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)

    def payload(self) -> dict[str, Any]:
        return json_safe(asdict(self))

    @property
    def edges(self) -> np.ndarray:
        return np.linspace(-self.usable_radius_m, self.usable_radius_m, self.primary_grid_bins + 1)

    @property
    def reference_edges(self) -> np.ndarray:
        return np.linspace(-self.usable_radius_m, self.usable_radius_m, self.reference_grid_bins + 1)

    @property
    def world_from_bucket_rotation(self) -> np.ndarray:
        return Rotation.from_euler("xyz", self.bucket_rpy_xyz_deg, degrees=True).as_matrix()

    @property
    def far_axis_bucket_xy(self) -> np.ndarray:
        far_world = self.bucket_center_world_m[:2] - self.robot_base_world_xy_m
        far_world /= np.linalg.norm(far_world)
        world3 = np.r_[far_world, 0.0]
        local3 = self.world_from_bucket_rotation.T @ world3
        axis = local3[:2]
        return axis / np.linalg.norm(axis)


def world_to_bucket(points_world: np.ndarray, config: DistributionConfig) -> np.ndarray:
    points = np.asarray(points_world, dtype=float)
    return (points - config.bucket_center_world_m) @ config.world_from_bucket_rotation


def bucket_polygon_mask(points_bucket: np.ndarray, config: DistributionConfig) -> np.ndarray:
    """Exact regular-polygon and height mask used by the reference project."""
    points = np.asarray(points_bucket, dtype=float)
    angles = (np.arange(config.bucket_segments) + 0.5) * 2.0 * np.pi / config.bucket_segments
    normals = np.stack([np.cos(angles), np.sin(angles)])
    apothem = config.bucket_radius_m * np.cos(np.pi / config.bucket_segments)
    horizontal = (points[:, :2] @ normals <= apothem).all(axis=1)
    floor = config.bucket_wall_m - config.voxel_size_m
    ceiling = config.bucket_height_m - 0.01
    return horizontal & (points[:, 2] >= floor) & (points[:, 2] < ceiling)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: tuple[float, ...]) -> np.ndarray:
    if len(values) == 0 or float(np.sum(weights)) <= 0.0:
        return np.zeros(len(quantiles), dtype=float)
    order = np.argsort(values)
    ordered = np.asarray(values, dtype=float)[order]
    ordered_weights = np.asarray(weights, dtype=float)[order]
    cumulative = np.cumsum(ordered_weights) - 0.5 * ordered_weights
    cumulative /= np.sum(ordered_weights)
    return np.interp(np.asarray(quantiles), cumulative, ordered, left=ordered[0], right=ordered[-1])


def _shape_score(points_xy: np.ndarray, weights: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> float:
    """Reference project's 64-direction whitened sliced-W2 score.

    The current particles have equal mass.  For nonuniform inputs, points are
    deterministically resampled before applying the original statistic.
    """
    if len(points_xy) <= 5:
        return 10.0
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues.min() <= 1.0e-12:
        return 10.0
    if not np.allclose(weights, weights[0]):
        cumulative = np.cumsum(weights) / np.sum(weights)
        samples = (np.arange(len(points_xy)) + 0.5) / len(points_xy)
        indices = np.searchsorted(cumulative, samples, side="left")
        points_xy = points_xy[np.clip(indices, 0, len(points_xy) - 1)]
    white = (points_xy - mean) @ eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues))
    angles = np.arange(64) * np.pi / 64
    directions = np.stack([np.cos(angles), np.sin(angles)])
    projected = np.sort(white @ directions, axis=0)
    target = norm.ppf((np.arange(len(points_xy)) + 0.5) / len(points_xy))[:, None]
    return float(np.sqrt(np.mean((projected - target) ** 2)))


def _region_mass(grid: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    sx = grid.shape[0] // shape[0]
    sy = grid.shape[1] // shape[1]
    return grid.reshape(shape[0], sx, shape[1], sy).sum(axis=(1, 3))


def _moments(points_xy: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(points_xy) == 0 or np.sum(weights) <= 0.0:
        return np.zeros(2), np.zeros((2, 2))
    normalized = weights / np.sum(weights)
    mean = normalized @ points_xy
    centered = points_xy - mean
    covariance = (centered * normalized[:, None]).T @ centered
    return mean, covariance


def _feature_vector(metrics: dict[str, Any], grid: np.ndarray, regions: np.ndarray, config: DistributionConfig) -> tuple[list[str], np.ndarray]:
    names = [f"grid_mass[{i:02d},{j:02d}]" for i in range(grid.shape[0]) for j in range(grid.shape[1])]
    values = list(grid.ravel())
    names.extend(f"region_mass[{i},{j}]" for i in range(regions.shape[0]) for j in range(regions.shape[1]))
    values.extend(regions.ravel())
    scalar_names = [
        "settled_usable_fraction", "inside_bucket_fraction", "moving_inside_fraction",
        "outside_bucket_fraction", "rim_annulus_fraction", "shovel_residual_fraction",
        "distribution_valid", "centroid_x_m", "centroid_y_m", "cov_xx_m2",
        "cov_xy_m2", "cov_yy_m2", "principal_std_major_m", "principal_std_minor_m",
        "principal_cos_2theta", "principal_sin_2theta", "occupied_cell_fraction",
        "normalized_entropy", "gaussian_shape_score", "final_speed_mean_m_s",
        "final_speed_p95_m_s", "centroid_drift_window_m", "stable_observation",
    ]
    scalar_values = [float(metrics[name]) for name in scalar_names]
    names.extend(scalar_names)
    values.extend(scalar_values)
    for axis_name in ("far", "tangent", "radial"):
        axis_values = metrics[f"{axis_name}_quantiles_m"]
        for quantile, value in zip(config.quantiles, axis_values):
            names.append(f"{axis_name}_q{int(round(100 * quantile)):02d}_m")
            values.append(float(value))
    vector = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=10.0, neginf=-10.0)
    return names, vector


def analyze_particle_distribution(
    particle_positions_world: np.ndarray,
    particle_velocities_world: np.ndarray,
    *,
    times_s: np.ndarray | None = None,
    retained_counts: np.ndarray | None = None,
    particle_masses_kg: np.ndarray | None = None,
    config: DistributionConfig | None = None,
) -> dict[str, Any]:
    """Analyze a final frame or a full replay and return a fixed schema."""
    config = config or DistributionConfig.load()
    positions = np.asarray(particle_positions_world, dtype=float)
    velocities = np.asarray(particle_velocities_world, dtype=float)
    if positions.ndim == 2:
        positions = positions[None, ...]
    if velocities.ndim == 2:
        velocities = velocities[None, ...]
    if positions.ndim != 3 or positions.shape[-1] != 3 or positions.shape != velocities.shape:
        raise ValueError("particle positions and velocities must be matching Nx3 or TxNx3 arrays")
    if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
        raise ValueError("particle replay contains non-finite values")
    frame_count, particle_count, _ = positions.shape
    if particle_count == 0:
        raise ValueError("particle replay is empty")
    masses = np.ones(particle_count, dtype=float) if particle_masses_kg is None else np.asarray(particle_masses_kg, dtype=float).reshape(particle_count)
    if np.any(masses < 0.0) or not np.isfinite(masses).all() or np.sum(masses) <= 0.0:
        raise ValueError("particle masses must be finite, nonnegative, and have positive total")
    mass_fraction = masses / np.sum(masses)
    local = world_to_bucket(positions[-1], config)
    speed = np.linalg.norm(velocities[-1], axis=1)
    inside = bucket_polygon_mask(local, config)
    radial = np.linalg.norm(local[:, :2], axis=1)
    usable = inside & (radial <= config.usable_radius_m)
    settled = usable & (speed <= config.settled_speed_m_s)
    moving_inside = inside & (speed > config.settled_speed_m_s)
    rim = inside & (radial > config.usable_radius_m)
    selected_xy = local[settled, :2]
    selected_weights = mass_fraction[settled]
    edges = config.edges
    grid, _, _ = np.histogram2d(
        selected_xy[:, 0], selected_xy[:, 1], bins=(edges, edges), weights=selected_weights,
    )
    reference_grid, reference_x_edges, reference_y_edges = np.histogram2d(
        selected_xy[:, 0], selected_xy[:, 1], bins=(config.reference_edges, config.reference_edges), weights=selected_weights,
    )
    regions = _region_mass(grid, config.region_grid_shape)
    mean, covariance = _moments(selected_xy, selected_weights)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    major_axis = eigenvectors[:, order[0]]
    angle = float(np.arctan2(major_axis[1], major_axis[0]))
    far_axis = config.far_axis_bucket_xy
    tangent_axis = np.asarray([-far_axis[1], far_axis[0]])
    far_values = selected_xy @ far_axis if len(selected_xy) else np.empty(0)
    tangent_values = selected_xy @ tangent_axis if len(selected_xy) else np.empty(0)
    radial_values = np.linalg.norm(selected_xy, axis=1) if len(selected_xy) else np.empty(0)
    conditional = grid / max(float(grid.sum()), 1.0e-12)
    positive = conditional[conditional > 0.0]
    entropy = float(-np.sum(positive * np.log(positive))) if len(positive) else 0.0
    centers = 0.5 * (edges[:-1] + edges[1:])
    xx, yy = np.meshgrid(centers, centers, indexing="ij")
    usable_cells = np.hypot(xx, yy) <= config.usable_radius_m
    occupied_fraction = float(np.count_nonzero(grid[usable_cells]) / max(1, np.count_nonzero(usable_cells)))
    normalized_entropy = entropy / max(np.log(max(2, np.count_nonzero(usable_cells))), 1.0)

    if times_s is None:
        times = np.arange(frame_count, dtype=float)
    else:
        times = np.asarray(times_s, dtype=float).reshape(frame_count)
        if np.any(np.diff(times) < 0.0):
            raise ValueError("times_s must be monotonic")
    window_start = times[-1] - config.stability_window_s
    window = np.flatnonzero(times >= window_start - 1.0e-12)
    first_window = int(window[0]) if len(window) else frame_count - 1
    first_centroid = np.mean(world_to_bucket(positions[first_window], config)[:, :2], axis=0)
    final_centroid_all = np.mean(local[:, :2], axis=0)
    centroid_drift = float(np.linalg.norm(final_centroid_all - first_centroid))
    final_speed_mean = float(np.sum(mass_fraction * speed))
    final_speed_p95 = float(_weighted_quantile(speed, mass_fraction, (0.95,))[0])
    residual_fraction = 0.0
    residual_count = 0
    minimum_retained_fraction = 0.0
    if retained_counts is not None:
        retained = np.asarray(retained_counts, dtype=float).reshape(frame_count)
        residual_count = int(round(retained[-1]))
        residual_fraction = float(retained[-1] / particle_count)
        minimum_retained_fraction = float(np.min(retained) / particle_count)
    distribution_valid = float(np.sum(selected_weights) > 0.0)
    stable_observation = float(
        centroid_drift <= config.stable_centroid_drift_m
        and final_speed_p95 <= config.stable_speed_p95_m_s
    )
    metrics: dict[str, Any] = {
        "particle_count": int(particle_count),
        "total_mass_kg": float(np.sum(masses)) if particle_masses_kg is not None else None,
        "settled_usable_count": int(np.sum(settled)),
        "settled_usable_fraction": float(np.sum(mass_fraction[settled])),
        "inside_bucket_count": int(np.sum(inside)),
        "inside_bucket_fraction": float(np.sum(mass_fraction[inside])),
        "moving_inside_count": int(np.sum(moving_inside)),
        "moving_inside_fraction": float(np.sum(mass_fraction[moving_inside])),
        "outside_bucket_count": int(np.sum(~inside)),
        "outside_bucket_fraction": float(np.sum(mass_fraction[~inside])),
        "rim_annulus_count": int(np.sum(rim)),
        "rim_annulus_fraction": float(np.sum(mass_fraction[rim])),
        "unrepresented_grid_fraction": float(max(0.0, 1.0 - np.sum(grid))),
        "shovel_residual_count": residual_count,
        "shovel_residual_fraction": residual_fraction,
        "minimum_retained_fraction": minimum_retained_fraction,
        "distribution_valid": distribution_valid,
        "centroid_bucket_xy_m": mean,
        "centroid_x_m": float(mean[0]),
        "centroid_y_m": float(mean[1]),
        "covariance_bucket_xy_m2": covariance,
        "cov_xx_m2": float(covariance[0, 0]),
        "cov_xy_m2": float(covariance[0, 1]),
        "cov_yy_m2": float(covariance[1, 1]),
        "principal_std_major_m": float(np.sqrt(eigenvalues[0])),
        "principal_std_minor_m": float(np.sqrt(eigenvalues[1])),
        "principal_angle_deg": float(np.rad2deg(angle) % 180.0),
        "principal_cos_2theta": float(np.cos(2.0 * angle)),
        "principal_sin_2theta": float(np.sin(2.0 * angle)),
        "far_quantiles_m": _weighted_quantile(far_values, selected_weights, config.quantiles),
        "tangent_quantiles_m": _weighted_quantile(tangent_values, selected_weights, config.quantiles),
        "radial_quantiles_m": _weighted_quantile(radial_values, selected_weights, config.quantiles),
        "near_half_fraction": float(np.sum(selected_weights[far_values <= 0.0])) if len(far_values) else 0.0,
        "far_half_fraction": float(np.sum(selected_weights[far_values > 0.0])) if len(far_values) else 0.0,
        "occupied_cell_fraction": occupied_fraction,
        "normalized_entropy": float(normalized_entropy),
        "effective_occupied_cells": float(np.exp(entropy)),
        "gaussian_shape_score": _shape_score(selected_xy, selected_weights, mean, covariance),
        "final_speed_mean_m_s": final_speed_mean,
        "final_speed_p95_m_s": final_speed_p95,
        "centroid_drift_window_m": centroid_drift,
        "stability_window_s": float(times[-1] - times[first_window]),
        "stable_observation": stable_observation,
        "observation_time_s": float(times[-1]),
    }
    names, vector = _feature_vector(metrics, grid, regions, config)
    return {
        "schema": "ec66_particle_distribution",
        "schema_version": SCHEMA_VERSION,
        "coordinate_frame": "bucket_local",
        "normalization": "each grid/region cell is mass divided by total particle mass",
        "grid_semantics": "settled particles inside the usable-radius disk; omitted mass is explicitly reported",
        "config": config.payload(),
        "metrics": json_safe(metrics),
        "mass_grid": grid,
        "reference_mass_grid_60": reference_grid,
        "x_edges_m": edges,
        "y_edges_m": edges.copy(),
        "reference_x_edges_m": reference_x_edges,
        "reference_y_edges_m": reference_y_edges,
        "region_mass": regions,
        "feature_names": names,
        "feature_vector": vector,
    }


def gaussian_distribution_goal(
    mean_bucket_xy_m: np.ndarray,
    covariance_bucket_xy_m2: np.ndarray,
    *,
    settled_fraction: float = 1.0,
    config: DistributionConfig | None = None,
) -> dict[str, Any]:
    """Create a quantized target distribution without running Newton."""
    config = config or DistributionConfig.load()
    mean = np.asarray(mean_bucket_xy_m, dtype=float).reshape(2)
    covariance = np.asarray(covariance_bucket_xy_m2, dtype=float).reshape(2, 2)
    if not 0.0 <= settled_fraction <= 1.0:
        raise ValueError("settled_fraction must lie in [0, 1]")
    if np.linalg.eigvalsh(covariance).min() <= 0.0:
        raise ValueError("Gaussian covariance must be positive definite")
    edges = config.edges
    centers = 0.5 * (edges[:-1] + edges[1:])
    xx, yy = np.meshgrid(centers, centers, indexing="ij")
    points = np.stack([xx, yy], axis=-1)
    delta = points - mean
    inverse = np.linalg.inv(covariance)
    density = np.exp(-0.5 * np.einsum("...i,ij,...j->...", delta, inverse, delta))
    density[np.hypot(xx, yy) > config.usable_radius_m] = 0.0
    if density.sum() <= 0.0:
        raise ValueError("Gaussian target has no support inside the usable pot disk")
    grid = settled_fraction * density / density.sum()
    regions = _region_mass(grid, config.region_grid_shape)
    flat_points = points.reshape(-1, 2)
    flat_weights = grid.ravel()
    selected = flat_weights > 0.0
    represented_points = flat_points[selected]
    represented_weights = flat_weights[selected]
    grid_mean, grid_covariance = _moments(represented_points, represented_weights)
    eigenvalues, eigenvectors = np.linalg.eigh(grid_covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    major = eigenvectors[:, order[0]]
    angle = float(np.arctan2(major[1], major[0]))
    far_axis = config.far_axis_bucket_xy
    tangent = np.asarray([-far_axis[1], far_axis[0]])
    far_values = represented_points @ far_axis
    tangent_values = represented_points @ tangent
    radial_values = np.linalg.norm(represented_points, axis=1)
    conditional = grid / max(grid.sum(), 1.0e-12)
    positive = conditional[conditional > 0.0]
    entropy = float(-np.sum(positive * np.log(positive)))
    usable_cells = np.hypot(xx, yy) <= config.usable_radius_m
    metrics = {
        "particle_count": None,
        "settled_usable_fraction": float(settled_fraction),
        "inside_bucket_fraction": float(settled_fraction),
        "moving_inside_fraction": 0.0,
        # Unrepresented target mass is a tolerance/unknown category, not a
        # request to throw grains outside the pot.
        "outside_bucket_fraction": 0.0,
        "rim_annulus_fraction": 0.0,
        "unrepresented_grid_fraction": float(1.0 - settled_fraction),
        "shovel_residual_fraction": 0.0,
        "distribution_valid": 1.0,
        "centroid_bucket_xy_m": grid_mean,
        "centroid_x_m": float(grid_mean[0]), "centroid_y_m": float(grid_mean[1]),
        "covariance_bucket_xy_m2": grid_covariance,
        "cov_xx_m2": float(grid_covariance[0, 0]), "cov_xy_m2": float(grid_covariance[0, 1]), "cov_yy_m2": float(grid_covariance[1, 1]),
        "principal_std_major_m": float(np.sqrt(eigenvalues[0])), "principal_std_minor_m": float(np.sqrt(eigenvalues[1])),
        "principal_angle_deg": float(np.rad2deg(angle) % 180.0),
        "principal_cos_2theta": float(np.cos(2.0 * angle)), "principal_sin_2theta": float(np.sin(2.0 * angle)),
        "far_quantiles_m": _weighted_quantile(far_values, represented_weights, config.quantiles),
        "tangent_quantiles_m": _weighted_quantile(tangent_values, represented_weights, config.quantiles),
        "radial_quantiles_m": _weighted_quantile(radial_values, represented_weights, config.quantiles),
        "occupied_cell_fraction": float(np.count_nonzero(grid[usable_cells]) / np.count_nonzero(usable_cells)),
        "normalized_entropy": entropy / max(np.log(max(2, np.count_nonzero(usable_cells))), 1.0),
        "gaussian_shape_score": 0.0,
        "final_speed_mean_m_s": 0.0, "final_speed_p95_m_s": 0.0,
        "centroid_drift_window_m": 0.0, "stable_observation": 1.0,
    }
    names, vector = _feature_vector(metrics, grid, regions, config)
    return {
        "schema": "ec66_particle_distribution", "schema_version": SCHEMA_VERSION,
        "role": "desired_distribution_goal", "coordinate_frame": "bucket_local",
        "normalization": "each grid/region cell is mass divided by total particle mass",
        "grid_semantics": "quantized desired settled mass; unrepresented mass is penalized",
        "config": config.payload(), "metrics": json_safe(metrics), "mass_grid": grid,
        "reference_mass_grid_60": np.zeros((config.reference_grid_bins, config.reference_grid_bins)),
        "x_edges_m": edges, "y_edges_m": edges.copy(),
        "reference_x_edges_m": config.reference_edges, "reference_y_edges_m": config.reference_edges,
        "region_mass": regions, "feature_names": names, "feature_vector": vector,
        "goal_specification": {"kind": "truncated_gaussian", "requested_mean_bucket_xy_m": mean, "requested_covariance_bucket_xy_m2": covariance, "settled_fraction": settled_fraction},
    }


def compare_distributions(actual: dict[str, Any], target: dict[str, Any], config: DistributionConfig | None = None) -> dict[str, float]:
    """Compute a bounded, decomposed loss suitable for inverse optimization."""
    config = config or DistributionConfig.load()
    a_grid = np.asarray(actual["mass_grid"], dtype=float)
    t_grid = np.asarray(target["mass_grid"], dtype=float)
    if a_grid.shape != t_grid.shape:
        raise ValueError(f"Distribution grid shape mismatch: {a_grid.shape} vs {t_grid.shape}")
    if actual.get("coordinate_frame") != target.get("coordinate_frame"):
        raise ValueError("Distribution coordinate frames do not match")
    if not np.allclose(np.asarray(actual["x_edges_m"]), np.asarray(target["x_edges_m"]), atol=1.0e-12):
        raise ValueError("Distribution x-grid edges do not match")
    if not np.allclose(np.asarray(actual["y_edges_m"]), np.asarray(target["y_edges_m"]), atol=1.0e-12):
        raise ValueError("Distribution y-grid edges do not match")
    a_metrics, t_metrics = actual["metrics"], target["metrics"]
    a_missing, t_missing = 1.0 - float(a_grid.sum()), 1.0 - float(t_grid.sum())
    grid_tv = 0.5 * (float(np.sum(np.abs(a_grid - t_grid))) + abs(a_missing - t_missing))
    region_tv = 0.5 * float(np.sum(np.abs(np.asarray(actual["region_mass"]) - np.asarray(target["region_mass"]))))
    dx = 2.0 * config.usable_radius_m / config.primary_grid_bins
    x_w1 = dx * float(np.sum(np.abs(np.cumsum(a_grid.sum(axis=1) - t_grid.sum(axis=1)))))
    y_w1 = dx * float(np.sum(np.abs(np.cumsum(a_grid.sum(axis=0) - t_grid.sum(axis=0)))))
    marginal_w1 = (x_w1 + y_w1) / (4.0 * config.usable_radius_m)
    a_mean = np.asarray(a_metrics["centroid_bucket_xy_m"], dtype=float)
    t_mean = np.asarray(t_metrics["centroid_bucket_xy_m"], dtype=float)
    centroid = float(np.linalg.norm(a_mean - t_mean) / (2.0 * config.usable_radius_m))
    a_cov = np.asarray(a_metrics["covariance_bucket_xy_m2"], dtype=float)
    t_cov = np.asarray(t_metrics["covariance_bucket_xy_m2"], dtype=float)
    covariance = float(np.linalg.norm(a_cov - t_cov, ord="fro") / (config.usable_radius_m ** 2))
    quantile_differences = []
    for axis in ("far", "tangent", "radial"):
        quantile_differences.extend(np.abs(np.asarray(a_metrics[f"{axis}_quantiles_m"]) - np.asarray(t_metrics[f"{axis}_quantiles_m"])))
    axis_quantiles = float(np.mean(quantile_differences) / (2.0 * config.usable_radius_m))
    categories = ("settled_usable_fraction", "moving_inside_fraction", "outside_bucket_fraction", "rim_annulus_fraction", "shovel_residual_fraction")
    category_fractions = float(np.mean([abs(float(a_metrics[name]) - float(t_metrics[name])) for name in categories]))
    components = {
        "grid_total_variation": grid_tv,
        "marginal_wasserstein": marginal_w1,
        "region_total_variation": region_tv,
        "centroid": centroid,
        "covariance": covariance,
        "axis_quantiles": axis_quantiles,
        "category_fractions": category_fractions,
    }
    total = sum(float(config.comparison_weights[name]) * value for name, value in components.items())
    return {**components, "weighted_total": float(total)}


def save_distribution_artifact(distribution: dict[str, Any], output_stem: Path, *, source: dict[str, Any] | None = None) -> tuple[Path, Path]:
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    npz_path = output_stem.with_suffix(".npz")
    json_path = output_stem.with_suffix(".json")
    arrays = {
        key: np.asarray(distribution[key]) for key in (
            "mass_grid", "reference_mass_grid_60", "x_edges_m", "y_edges_m",
            "reference_x_edges_m", "reference_y_edges_m", "region_mass", "feature_vector",
        )
    }
    arrays["feature_names"] = np.asarray(distribution["feature_names"], dtype=np.str_)
    np.savez_compressed(npz_path, **arrays)
    digest = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    payload = {
        key: json_safe(value) for key, value in distribution.items()
        if key not in arrays and key != "feature_names"
    }
    payload.update({"array_file": str(npz_path.resolve()), "array_sha256": digest, "feature_count": len(distribution["feature_names"]), "source": json_safe(source or {})})
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return npz_path, json_path


def load_distribution_artifact(path: Path) -> dict[str, Any]:
    path = Path(path)
    json_path = path if path.suffix == ".json" else path.with_suffix(".json")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    npz_path = Path(payload["array_file"])
    if not npz_path.is_file():
        npz_path = json_path.with_suffix(".npz")
    with np.load(npz_path, allow_pickle=False) as stored:
        for key in stored.files:
            payload[key] = np.asarray(stored[key])
    payload["feature_names"] = [str(value) for value in payload["feature_names"]]
    return payload


def write_distribution_svg(distribution: dict[str, Any], path: Path, title: str = "Particle distribution") -> None:
    """Write a dependency-free heatmap for quick result inspection."""
    grid = np.asarray(distribution["mass_grid"], dtype=float)
    metrics = distribution["metrics"]
    size, margin = 720, 70
    panel = size - 2 * margin
    maximum = max(float(grid.max()), 1.0e-12)
    cells = []
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            value = np.log1p(1000.0 * grid[i, j]) / np.log1p(1000.0 * maximum)
            red = int(35 + 220 * value)
            green = int(45 + 150 * np.sqrt(value))
            blue = int(90 - 65 * value)
            x = margin + i * panel / grid.shape[0]
            y = margin + (grid.shape[1] - 1 - j) * panel / grid.shape[1]
            cells.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{panel/grid.shape[0]+0.2:.2f}" height="{panel/grid.shape[1]+0.2:.2f}" fill="rgb({red},{green},{blue})"/>')
    center = margin + panel / 2
    radius = panel / 2
    centroid = np.asarray(metrics["centroid_bucket_xy_m"], dtype=float)
    cx = center + centroid[0] / float(distribution["config"]["usable_radius_m"]) * radius
    cy = center - centroid[1] / float(distribution["config"]["usable_radius_m"]) * radius
    subtitle = (
        f"settled={100*float(metrics['settled_usable_fraction']):.1f}%  "
        f"outside={100*float(metrics['outside_bucket_fraction']):.1f}%  "
        f"residual={100*float(metrics['shovel_residual_fraction']):.1f}%  "
        f"stable={'yes' if float(metrics['stable_observation']) else 'no'}"
    )
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size+70}" viewBox="0 0 {size} {size+70}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{size/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="22">{title}</text>',
        *cells,
        f'<circle cx="{center}" cy="{center}" r="{radius}" fill="none" stroke="#e2e8f0" stroke-width="3"/>',
        f'<line x1="{cx-9:.2f}" y1="{cy:.2f}" x2="{cx+9:.2f}" y2="{cy:.2f}" stroke="white" stroke-width="3"/>',
        f'<line x1="{cx:.2f}" y1="{cy-9:.2f}" x2="{cx:.2f}" y2="{cy+9:.2f}" stroke="white" stroke-width="3"/>',
        f'<text x="{size/2}" y="{size+35}" text-anchor="middle" font-family="sans-serif" font-size="15">{subtitle}</text>',
        '</svg>',
    ]
    Path(path).write_text("\n".join(svg), encoding="utf-8")
