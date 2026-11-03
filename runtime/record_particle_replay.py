"""Run the shared Near_Constrained Newton scene and save a 60 Hz replay."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import newton.examples
import numpy as np


from release_reference_viewer import ReleaseExample  # noqa: E402


class RecordingReleaseExample(ReleaseExample):
    def __init__(self, viewer, args):
        self.replay_output = Path(args.replay_output).resolve()
        self.replay_saved = False
        self.replay_times: list[float] = []
        self.replay_q: list[np.ndarray] = []
        self.replay_v: list[np.ndarray] = []
        self.replay_tcp: list[np.ndarray] = []
        self.replay_particles: list[np.ndarray] = []
        self.replay_particle_velocities: list[np.ndarray] = []
        self.replay_retained: list[int] = []
        super().__init__(viewer, args)
        self._record_frame()

    def _record_frame(self) -> None:
        if self.replay_times and abs(self.sim_time - self.replay_times[-1]) < 1.0e-10:
            return
        self.replay_times.append(float(self.sim_time))
        self.replay_q.append(np.asarray(self.q, dtype=np.float32).copy())
        self.replay_v.append(
            np.asarray(self.scene.qd_host[self.scene.joint_dof_ids], dtype=np.float32).copy()
        )
        self.replay_tcp.append(np.asarray(self.target, dtype=np.float32).copy())
        self.replay_particles.append(
            self.scene.sand_state.particle_q.numpy().astype(np.float32, copy=True)
        )
        self.replay_particle_velocities.append(
            self.scene.sand_state.particle_qd.numpy().astype(np.float32, copy=True)
        )
        self.replay_retained.append(int(np.sum(self.was_inside)))

    def _save_replay(self) -> None:
        if self.replay_saved:
            return
        self.replay_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.replay_output,
            trajectory=str(self.trajectory_path),
            times_s=np.asarray(self.replay_times, dtype=np.float64),
            q=np.asarray(self.replay_q, dtype=np.float32),
            v=np.asarray(self.replay_v, dtype=np.float32),
            tcp_positions=np.asarray(self.replay_tcp, dtype=np.float32),
            particle_positions=np.asarray(self.replay_particles, dtype=np.float32),
            particle_velocities=np.asarray(self.replay_particle_velocities, dtype=np.float32),
            retained_counts=np.asarray(self.replay_retained, dtype=np.int32),
            initial_hold_seconds=float(self.initial_hold),
            motion_duration_s=float(self.motion_duration),
            physics_substeps=int(self.scene.cfg["substeps"]),
        )
        self.replay_saved = True
        print(f"Replay: {self.replay_output}", flush=True)

    def step(self) -> None:
        previous_time = float(self.sim_time)
        super().step()
        if self.sim_time > previous_time + 1.0e-12:
            self._record_frame()
        if self.sim_time >= self.total_duration - 1.0e-10:
            self._save_replay()

    def test_final(self) -> None:
        super().test_final()
        self._record_frame()
        self._save_replay()


def main() -> None:
    parser = newton.examples.create_parser()
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--departure-output", required=True)
    parser.add_argument("--replay-output", required=True)
    parser.add_argument("--post-settle-seconds", type=float, default=0.0)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--trajectory-start-s", type=float)
    parser.add_argument("--trajectory-end-s", type=float)
    parser.add_argument("--initial-hold-seconds", type=float, default=0.25)
    parser.add_argument("--autoplay", action="store_true")
    parser.set_defaults(coverage_only=False, capture=None)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(RecordingReleaseExample(viewer, args), args)


if __name__ == "__main__":
    main()
