"""Record the shared Newton scene, without resetting material at release."""
from __future__ import annotations

from argparse import Namespace
import hashlib
from pathlib import Path

import numpy as np

from bootstrap import RUNTIME
from design import read_json, write_json
from motion import load_motion


def simulate(trajectory, directory, spec, context, *, ideal, release_time=0., status=lambda m: None):
    # Deferred imports keep planning/GUI startup independent of the particle scene.
    import warp as wp
    from newton.viewer import ViewerNull
    import newton
    import scipy
    from record_particle_replay import RecordingReleaseExample
    from particle_distribution import analyze_particle_distribution, save_distribution_artifact

    class Simulation(RecordingReleaseExample):
        def __init__(self, viewer, args):
            self.ideal = ideal
            self.exact_release_time = release_time
            super().__init__(viewer, args)
            if ideal:
                points = self.scene.sand_state.particle_q.numpy()
                pose = context.kinematics.tool_pose_world(self.q, context.tool_reference)
                velocity = context.kinematics.tool_twist_world(self.q, self.smooth(0, 1), context.tool_reference)
                values = velocity[:3] + np.cross(velocity[3:], points - pose[:3, 3])
            else:
                values = np.zeros_like(self.scene.sand_state.particle_q.numpy())
            if spec["simulation"]["load_mode"] == "velocity_jitter":
                rng = np.random.default_rng(int(spec["simulation"]["seed"]))
                values += rng.normal(0., spec["simulation"]["velocity_noise_m_s"], values.shape)
            self.scene.sand_state.particle_qd.assign(values.astype(np.float32))
            self.previous_particle_velocity = values.astype(np.float32)
            self.replay_particle_velocities[-1] = values.astype(np.float32)
            self.replay_v[-1] = np.asarray(self.smooth(0, 1), dtype=np.float32)
            self.update()

        def step(self):
            if not self.playing:
                return
            frame_end = min(self.total_duration, self.sim_time + 1 / 60)
            while self.sim_time < frame_end - 1e-12:
                dt = min(1 / 60 / self.scene.cfg["substeps"], frame_end - self.sim_time)
                if self.sim_time + 1e-10 < self.exact_release_time < self.sim_time + dt:
                    dt = self.exact_release_time - self.sim_time
                self.sim_time += dt
                self.update()
                if self.grains is None:
                    raise RuntimeError("This workbench requires the existing XPBD granular scene")
                self.grains.step(dt)
                self.scene.sand_state = self.grains.a
                self.physics_steps += 1
                self._track_departure(dt)
                if abs(self.sim_time - self.exact_release_time) < 1e-9:
                    self._record_frame()
            self._record_frame()
            if self.sim_time >= self.total_duration - 1e-10:
                self.playing = False
                self._save_departure()
                self._save_replay()

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    args = Namespace(trajectory=str(Path(trajectory).resolve()), departure_output=str(directory / "departures.npz"),
                     replay_output=str(directory / "replay.npz"), coverage_only=False, capture=None,
                     autoplay=True, post_settle_seconds=spec["simulation"]["settle_seconds"],
                     substeps=spec["simulation"]["substeps"], initial_hold_seconds=0.,
                     trajectory_start_s=None, trajectory_end_s=None)
    status("Initializing Newton particles (first run may compile GPU kernels)")
    sim = Simulation(ViewerNull(), args)
    frames = 0
    while sim.playing:
        sim.step()
        frames += 1
        if frames % 60 == 0:
            status(f"Newton {sim.sim_time:.1f}/{sim.total_duration:.1f}s")
    wp.synchronize()
    positions = np.asarray(sim.replay_particles)
    velocities = np.asarray(sim.replay_particle_velocities)
    if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
        raise ValueError("Newton produced non-finite particles")
    times = np.asarray(sim.replay_times)
    retained = np.asarray(sim.replay_retained)
    distribution = analyze_particle_distribution(positions, velocities, times_s=times, retained_counts=retained)
    scene_config = RUNTIME / "scene_config_particles.json"
    load_snapshot = RUNTIME / read_json(scene_config)["particle_positions_file"]
    save_distribution_artifact(distribution, directory / "distribution", source={
        "trajectory": str(Path(trajectory).resolve()), "initialization": "ideal_rigid_carried" if ideal else "fixed_start_continuous",
        "trajectory_sha256": hashlib.sha256(Path(trajectory).read_bytes()).hexdigest(),
        "physics_substeps": spec["simulation"]["substeps"], "load_mode": spec["simulation"]["load_mode"],
        "particle_scene_sha256": hashlib.sha256(scene_config.read_bytes()).hexdigest(),
        "particle_load_sha256": hashlib.sha256(load_snapshot.read_bytes()).hexdigest(),
        "runtime": {"newton": getattr(newton, "__version__", "unknown"), "warp": wp.__version__,
                    "numpy": np.__version__, "scipy": scipy.__version__, "device": str(wp.get_device())}})
    release_index = int(np.argmin(abs(times - release_time)))
    motion = load_motion(trajectory)
    q = motion(release_time)
    pose = context.kinematics.tool_pose_world(q, context.tool_reference)
    twist = context.kinematics.tool_twist_world(q, motion(release_time, 1), context.tool_reference)
    local = (positions[release_index] - pose[:3, 3]) @ pose[:3, :3] + context.tool_reference
    relative = velocities[release_index] - twist[:3] - np.cross(twist[3:], positions[release_index] - pose[:3, 3])
    mesh = np.asarray(sim.scene.shovel_mesh.vertices)
    inside = np.all((local >= mesh.min(axis=0) - .02) & (local <= mesh.max(axis=0) + .02), axis=1)
    speeds = np.linalg.norm(relative[inside], axis=1)
    pre = times <= release_time + 1e-9
    result = {"initialization": "ideal_rigid_carried" if ideal else "fixed_start_continuous",
              "release_time_s": release_time, "release_frame_time_s": float(times[release_index]),
              "particle_count": int(positions.shape[1]),
              "release_retained_fraction": float(inside.mean()),
              "minimum_pre_release_retention": float(retained[pre].min() / positions.shape[1]),
              "release_relative_speed_rms_m_s": float(np.sqrt(np.mean(speeds ** 2))) if len(speeds) else None,
              "release_relative_speed_p95_m_s": float(np.quantile(speeds, .95)) if len(speeds) else None,
              "metrics": distribution["metrics"],
              "load_mode": spec["simulation"]["load_mode"],
              "retention_definition": "instantaneous shovel mesh AABB expanded by 20mm; sampled at 60Hz plus exact release"}
    write_json(directory / "result.json", result)
    np.savez_compressed(directory / "release_particles.npz", positions=positions[release_index],
                        velocities=velocities[release_index], relative_velocities=relative,
                        shovel_positions=local, inside=inside, q=q, qd=motion(release_time, 1), qdd=motion(release_time, 2))
    return result
