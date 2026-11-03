"""Newton viewer for continuous release trajectories with particle exit logging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import newton
import newton.examples
import numpy as np
import warp as wp
from scipy.interpolate import PPoly
from scipy.spatial.transform import Rotation

from particle_scene import Example as BaseExample, ROOT
from distribution_overlay import CFG, gaussian_w2, pot_mask


GAUSSIAN_LEVELS = (
    ("68%", 1.5095921854516636),
    ("95%", 2.447746830680816),
)


def _goal_covariance(goal):
    goal = np.asarray(goal, dtype=float)
    cholesky = np.array(
        [[np.exp(goal[2]), 0.0], [goal[3], np.exp(goal[4])]],
        dtype=float,
    )
    return cholesky @ cholesky.T


def _ellipse_segments(mean, covariance, z):
    """Return Warp line segments for the 68% and 95% Gaussian contours."""
    values, vectors = np.linalg.eigh(np.asarray(covariance, dtype=float))
    values = np.maximum(values, 1e-12)
    angles = np.linspace(0.0, 2.0 * np.pi, 129)
    unit_circle = np.column_stack((np.cos(angles), np.sin(angles)))
    transform = vectors @ np.diag(np.sqrt(values))
    contours = []
    for _label, scale in GAUSSIAN_LEVELS:
        xy = np.asarray(mean, dtype=float) + scale * (unit_circle @ transform.T)
        xyz = np.column_stack((xy, np.full(len(xy), z))).astype(np.float32)
        contours.append(
            (
                wp.array(xyz[:-1], dtype=wp.vec3),
                wp.array(xyz[1:], dtype=wp.vec3),
            )
        )
    return contours


class ReleaseExample(BaseExample):
    def __init__(self, viewer, args):
        self.trajectory_path = Path(args.trajectory).resolve()
        self.departure_output = Path(args.departure_output).resolve() if args.departure_output else None
        self.coverage_only = bool(args.coverage_only)
        self.saved_departure = False
        self.gaussian_goal = None
        self.target_covariance = None
        self.target_contours = []
        self.observed_contours = []
        self.target_mean_point = None
        self.observed_mean_point = None
        self.live_distribution = None
        self._last_distribution_frame = -1000
        super().__init__(viewer, args)
        if args.substeps is not None:
            if args.substeps < 1:
                raise ValueError("--substeps must be at least 1")
            self.scene.cfg["substeps"] = int(args.substeps)

    def load(self, index):
        # BaseExample.__init__ first loads an ordinary student path.
        path = self.files[index]
        with np.load(path, allow_pickle=False) as data:
            if "coefficients" not in data:
                self._clear_gaussian_overlay()
                self.smooth = None
                super().load(index)
                self.source_start = 0.0
                self.source_end = self.duration
                self.initial_hold = 0.0
                self.motion_duration = self.duration
                self.release_start = 0.0
                self.total_duration = self.duration + max(0.0, float(self.args.post_settle_seconds))
                self._reset_departure_tracking()
                return
            self.index = index
            self.qs = np.asarray(data["q"], dtype=float)
            self.poses = np.asarray(data["poses"], dtype=float)
            self.times = np.asarray(data["t"], dtype=float)
            self.smooth = PPoly(np.asarray(data["coefficients"], dtype=float), self.times)
            full_duration = float(self.times[-1])
            requested_start = (
                0.0
                if self.args.trajectory_start_s is None
                else float(self.args.trajectory_start_s)
            )
            requested_end = (
                full_duration
                if self.args.trajectory_end_s is None
                else float(self.args.trajectory_end_s)
            )
            self.source_start = float(np.clip(requested_start, 0.0, full_duration))
            self.source_end = float(np.clip(requested_end, self.source_start, full_duration))
            self.initial_hold = max(0.0, float(self.args.initial_hold_seconds))
            self.motion_duration = self.source_end - self.source_start
            self.duration = self.initial_hold + self.motion_duration
            self.total_duration = self.duration + max(0.0, float(self.args.post_settle_seconds))
            raw_release = float(data["release_start_s"]) if "release_start_s" in data else 0.0
            self.release_start = self.initial_hold + float(
                np.clip(raw_release - self.source_start, 0.0, self.motion_duration)
            )
            if "gaussian_goal_vector" in data:
                self._set_gaussian_goal(np.asarray(data["gaussian_goal_vector"], dtype=float))
            else:
                self._clear_gaussian_overlay()
            # BaseExample.reset_particles() uses qs[0] to place the shovel and grains.
            # For a cropped run this must be the crop's first pose, not the full path's.
            self.qs[0] = np.asarray(self.smooth(self.source_start), dtype=float)
        self.sim_time = 0.0
        self.last = 0.0
        self.update()
        self.reset_particles()
        self._reset_departure_tracking()

    def _clear_gaussian_overlay(self):
        self.gaussian_goal = None
        self.target_covariance = None
        self.target_contours = []
        self.observed_contours = []
        self.target_mean_point = None
        self.observed_mean_point = None
        self.live_distribution = None
        self._last_distribution_frame = -1000

    def _set_gaussian_goal(self, goal):
        if goal.shape != (5,) or not np.isfinite(goal).all():
            raise ValueError("gaussian_goal_vector must contain five finite values")
        self.gaussian_goal = goal.copy()
        self.target_covariance = _goal_covariance(goal)
        floor_z = float(CFG["bucket_xyz"][2]) + float(CFG["bucket_wall"])
        target_z = floor_z + 0.012
        self.target_contours = _ellipse_segments(goal[:2], self.target_covariance, target_z)
        self.target_mean_point = wp.array(
            [[goal[0], goal[1], target_z + 0.006]], dtype=wp.vec3
        )
        self.observed_contours = []
        self.observed_mean_point = None
        self.live_distribution = None
        self._last_distribution_frame = -1000

    def _update_distribution_overlay(self, force=False):
        if self.gaussian_goal is None:
            return
        # Fast preview avoids frequent GPU-to-CPU synchronization.  Validation
        # and diagnostic modes retain the denser ten-frame overlay refresh.
        refresh_frames = 30 if self.coverage_only else 10
        if not force and self.frames - self._last_distribution_frame < refresh_frames:
            return
        self._last_distribution_frame = self.frames
        particles = self.scene.sand_state.particle_q.numpy()
        inside = pot_mask(particles)
        xy = np.asarray(particles[inside, :2], dtype=float)
        if len(xy) <= 5:
            self.observed_contours = []
            self.observed_mean_point = None
            self.live_distribution = {
                "inside_count": int(len(xy)),
                "inside_fraction": float(inside.mean()),
            }
            return
        mean = xy.mean(axis=0)
        covariance = np.cov(xy, rowvar=False, bias=True)
        floor_z = float(CFG["bucket_xyz"][2]) + float(CFG["bucket_wall"])
        observed_z = floor_z + 0.018
        self.observed_contours = _ellipse_segments(mean, covariance, observed_z)
        self.observed_mean_point = wp.array(
            [[mean[0], mean[1], observed_z + 0.006]], dtype=wp.vec3
        )
        self.live_distribution = {
            "inside_count": int(len(xy)),
            "inside_fraction": float(inside.mean()),
            "mean": mean,
            "covariance": covariance,
            "center_error_m": float(np.linalg.norm(mean - self.gaussian_goal[:2])),
            "gaussian_w2_m": gaussian_w2(mean, covariance, self.gaussian_goal),
        }

    def _reset_departure_tracking(self):
        count = int(self.scene.sand_model.particle_count)
        self.was_inside = np.ones(count, dtype=bool)
        # time, position xyz, velocity xyz, acceleration xyz, speed, acceleration norm
        self.departure = np.full((count, 12), np.nan, dtype=float)
        self.previous_particle_velocity = (
            None if self.coverage_only else self.scene.sand_state.particle_qd.numpy().copy()
        )
        self.saved_departure = False

    def update(self):
        if getattr(self, "smooth", None) is None:
            elapsed = self.sim_time
            self.sim_time = min(elapsed, self.duration)
            super().update()
            self.sim_time = elapsed
            if elapsed >= self.duration:
                self.scene.qd_host.fill(0)
                self.scene.state.joint_qd.assign(self.scene.qd_host)
                newton.eval_fk(
                    self.scene.model,
                    self.scene.state.joint_q,
                    self.scene.state.joint_qd,
                    self.scene.state,
                )
            return
        motion_elapsed = max(0.0, self.sim_time - self.initial_hold)
        t = min(self.source_start + motion_elapsed, self.source_end)
        self.q = np.asarray(self.smooth(t), dtype=float)
        moving = self.playing and self.initial_hold <= self.sim_time < self.duration
        qd = np.asarray(self.smooth(t, 1), dtype=float) if moving else np.zeros(6)
        self.scene.q_host[self.scene.joint_q_ids] = self.q
        self.scene.qd_host.fill(0)
        self.scene.qd_host[self.scene.joint_dof_ids] = qd
        self.scene.state.joint_q.assign(self.scene.q_host)
        self.scene.state.joint_qd.assign(self.scene.qd_host)
        newton.eval_fk(
            self.scene.model,
            self.scene.state.joint_q,
            self.scene.state.joint_qd,
            self.scene.state,
        )
        position, _ = self.arm.fk(self.q)
        self.target = self.W[:3, :3] @ position + self.W[:3, 3]
        self.error = (0.0, 0.0)
        self.current_segment = int(np.clip(np.searchsorted(self.times, t, side="right") - 1, 0, len(self.times) - 2))

    def _track_departure(self, dt):
        if self.coverage_only:
            return
        particles = self.scene.sand_state.particle_q.numpy()
        velocity = self.scene.sand_state.particle_qd.numpy()
        acceleration = (velocity - self.previous_particle_velocity) / dt
        flange = self.scene.state.body_q.numpy()[self.scene.flange_id]
        rotation = Rotation.from_quat(flange[3:]).as_matrix() @ self.scene.mount[:3, :3]
        origin = flange[:3] + Rotation.from_quat(flange[3:]).as_matrix() @ self.scene.mount[:3, 3]
        local = (particles - origin) @ rotation
        mesh = np.asarray(self.scene.shovel_mesh.vertices)
        inside = np.all((local >= mesh.min(axis=0) - 0.02) & (local <= mesh.max(axis=0) + 0.02), axis=1)
        exits = self.was_inside & ~inside
        if np.any(exits):
            self.departure[exits] = np.column_stack(
                [
                    np.full(int(exits.sum()), self.sim_time),
                    particles[exits],
                    velocity[exits],
                    acceleration[exits],
                    np.linalg.norm(velocity[exits], axis=1),
                    np.linalg.norm(acceleration[exits], axis=1),
                ]
            )
        # A re-entering grain has not made its final departure yet.
        self.departure[inside] = np.nan
        self.was_inside = inside
        self.previous_particle_velocity = velocity.copy()

    def _save_departure(self):
        if self.departure_output is None or self.saved_departure:
            return
        self.departure_output.parent.mkdir(parents=True, exist_ok=True)
        final = self.scene.sand_state.particle_q.numpy().copy()
        velocity = self.scene.sand_state.particle_qd.numpy().copy()
        valid = np.isfinite(self.departure[:, 0])
        early = valid & (self.departure[:, 0] < self.release_start)
        np.savez_compressed(
            self.departure_output,
            trajectory=str(self.trajectory_path),
            release_start_s=self.release_start,
            trajectory_source_window_s=np.asarray([self.source_start, self.source_end]),
            initial_hold_seconds=self.initial_hold,
            departure=self.departure,
            final=final,
            final_velocity=velocity,
            departed=valid,
            early=early,
            remaining=~valid,
        )
        report = {
            "trajectory": str(self.trajectory_path),
            "release_start_s": self.release_start,
            "trajectory_source_window_s": [self.source_start, self.source_end],
            "initial_hold_seconds": self.initial_hold,
            "coverage_only": self.coverage_only,
            "substeps": int(self.scene.cfg["substeps"]),
            "departed_count": int(valid.sum()),
            "early_departure_count": int(early.sum()),
            "remaining_count": int((~valid).sum()),
            "departure_speed_mean_m_s": float(np.mean(self.departure[valid, 10])) if valid.any() else None,
            "departure_speed_p90_m_s": float(np.percentile(self.departure[valid, 10], 90)) if valid.any() else None,
            "departure_acceleration_mean_m_s2": float(np.mean(self.departure[valid, 11])) if valid.any() else None,
            "departure_acceleration_p90_m_s2": float(np.percentile(self.departure[valid, 11], 90)) if valid.any() else None,
            "criterion": (
                "disabled in coverage-only mode"
                if self.coverage_only
                else "final exit from shovel-local mesh AABB expanded by 0.02 m"
            ),
        }
        self.departure_output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps(report, ensure_ascii=False), flush=True)
        self.saved_departure = True

    def step(self):
        if not self.playing:
            return
        scene = self.scene
        dt = 1 / 60 / scene.cfg["substeps"]
        for _ in range(scene.cfg["substeps"]):
            if self.sim_time >= self.total_duration:
                break
            self.sim_time = min(self.total_duration, self.sim_time + dt)
            self.update()
            if self.grains is not None:
                self.grains.step(dt)
                scene.sand_state = self.grains.a
            else:
                scene.sand_state.body_q.assign(scene.state.body_q)
                scene.sand_state.body_qd.assign(scene.state.body_qd)
                scene.mpm_solver.step(scene.sand_state, scene.sand_state, contacts=None, control=None, dt=dt)
                scene.mpm_solver.project_outside(
                    scene.sand_state,
                    scene.sand_state,
                    dt=dt,
                    gap=4 * scene.cfg["voxel_size"],
                )
            self.physics_steps += 1
            self._track_departure(dt)
        if self.sim_time >= self.total_duration:
            self.playing = False
            self._update_distribution_overlay(force=True)
            self._save_departure()

    def render(self):
        self._update_distribution_overlay()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.scene.state)
        self.viewer.log_points(
            "Sticky material",
            self.scene.sand_state.particle_q,
            self.scene.sand_model.particle_radius,
            self.scene.particle_colors,
        )
        world = self.poses[:, :3] @ self.W[:3, :3].T + self.W[:3, 3]
        self.viewer.log_lines(
            "Recorded path",
            wp.array(world[:-1], dtype=wp.vec3),
            wp.array(world[1:], dtype=wp.vec3),
            (0.1, 0.8, 1.0),
        )
        self.viewer.log_points(
            "Target flange",
            wp.array([self.target], dtype=wp.vec3),
            0.012,
            wp.array([(1.0, 0.15, 0.1)], dtype=wp.vec3),
        )

        if self.target_contours:
            self.viewer.log_lines(
                "Target Gaussian 68%",
                *self.target_contours[0],
                (0.75, 0.25, 1.0),
            )
            self.viewer.log_lines(
                "Target Gaussian 95%",
                *self.target_contours[1],
                (1.0, 0.18, 0.12),
            )
            self.viewer.log_points(
                "Target Gaussian mean",
                self.target_mean_point,
                0.018,
                wp.array([(1.0, 0.35, 0.08)], dtype=wp.vec3),
            )
        else:
            self.viewer.log_lines("Target Gaussian 68%", None, None, None)
            self.viewer.log_lines("Target Gaussian 95%", None, None, None)

        if self.observed_contours:
            self.viewer.log_lines(
                "Observed particles 68%",
                *self.observed_contours[0],
                (0.0, 0.95, 1.0),
            )
            self.viewer.log_lines(
                "Observed particles 95%",
                *self.observed_contours[1],
                (1.0, 0.85, 0.05),
            )
            self.viewer.log_points(
                "Observed particle mean",
                self.observed_mean_point,
                0.014,
                wp.array([(0.15, 1.0, 0.2)], dtype=wp.vec3),
            )
        else:
            self.viewer.log_lines("Observed particles 68%", None, None, None)
            self.viewer.log_lines("Observed particles 95%", None, None, None)

        self.viewer.end_frame()
        self.frames += 1
        if self.args.capture and self.frames == 60:
            import pyglet

            pyglet.image.get_buffer_manager().get_color_buffer().save(self.args.capture)

    def gui(self, ui):
        ui.text("C2 RELEASE TRAJECTORY | Lite project seed")
        ui.text(self.trajectory_path.name)
        if ui.button("Pause" if self.playing else "Play"):
            self.playing = not self.playing
        ui.same_line()
        if ui.button("Restart"):
            self.load(0)
            self.playing = True
        valid = np.isfinite(self.departure[:, 0])
        ui.text(f"Time: {self.sim_time:.3f} / {self.total_duration:.3f} s")
        ui.text(f"Motion end: {self.duration:.3f} s | settle: {self.total_duration - self.duration:.3f} s")
        ui.text(f"Source window: {self.source_start:.3f} .. {self.source_end:.3f} s")
        ui.text(f"Planned release: {self.release_start:.3f} s")
        ui.text(f"Physics substeps: {self.scene.cfg['substeps']}")
        if self.coverage_only:
            ui.text("Coverage-only: departure tracking disabled")
        else:
            ui.text(f"Departed: {int(valid.sum())} | Remaining: {int((~valid).sum())}")
            ui.text("Departure = final exit from shovel AABB (+20 mm)")
        if self.gaussian_goal is not None:
            values, vectors = np.linalg.eigh(self.target_covariance)
            major = int(np.argmax(values))
            angle = np.degrees(np.arctan2(vectors[1, major], vectors[0, major]))
            ui.text("Gaussian overlay (pot bottom)")
            ui.text("Target: purple 68% | red 95% | orange center")
            ui.text("Observed: cyan 68% | yellow 95% | green center")
            ui.text(
                f"Target mean: ({self.gaussian_goal[0]:+.3f}, "
                f"{self.gaussian_goal[1]:+.3f}) m"
            )
            ui.text(
                f"Target sigma: {np.sqrt(values[major]) * 100:.1f} x "
                f"{np.sqrt(values[1 - major]) * 100:.1f} cm | angle {angle:+.1f} deg"
            )
            if self.live_distribution is not None:
                live = self.live_distribution
                ui.text(
                    f"Inside pot: {live['inside_count']}/{len(self.was_inside)} "
                    f"({live['inside_fraction'] * 100:.1f}%)"
                )
                if "center_error_m" in live:
                    ui.text(f"Center error: {live['center_error_m'] * 100:.2f} cm")
                    ui.text(f"Gaussian W2: {live['gaussian_w2_m'] * 100:.2f} cm")

    def test_final(self):
        if not np.isfinite(self.scene.state.body_q.numpy()).all():
            raise AssertionError("Non-finite robot state")
        if not np.isfinite(self.scene.sand_state.particle_q.numpy()).all():
            raise AssertionError("Non-finite particle state")
        self._save_departure()


def main():
    parser = newton.examples.create_parser()
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--departure-output")
    parser.add_argument("--post-settle-seconds", type=float, default=1.5)
    parser.add_argument("--coverage-only", action="store_true")
    parser.add_argument("--substeps", type=int)
    parser.add_argument("--trajectory-start-s", type=float)
    parser.add_argument("--trajectory-end-s", type=float)
    parser.add_argument("--initial-hold-seconds", type=float, default=0.0)
    parser.add_argument("--capture")
    parser.add_argument("--autoplay", action="store_true")
    viewer, args = newton.examples.init(parser)
    example = ReleaseExample(viewer, args)
    newton.examples.run(example, args)


if __name__ == "__main__":
    main()
