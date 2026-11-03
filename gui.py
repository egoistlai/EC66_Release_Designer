"""Native Newton/ImGui release editor with asynchronous planning and replay."""
from __future__ import annotations

import copy
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp
import newton.examples

from bootstrap import ROOT
from design import (default_design, validate, read_json, write_json, fingerprint,
                    fixed_start, save_design, create_run)
from motion import load_motion
from full_state_controller import FullStateExample


class Designer(FullStateExample):
    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        self.full_cfg["joint_velocity_design_fraction"] = .95
        self.spec = read_json(args.scheme) if args.scheme else default_design()
        self.live_solve = True
        self.release_dirty = False
        self.release_solve_error = None
        self.show_target_velocity = True
        self.show_solved_velocity = True
        self.process = None
        self.process_log = None
        self.run_dir = None
        self.run_hash = None
        self.job = {}
        self.result = None
        self.audit_report = None
        self.samples = None
        self.replay = None
        self.distribution = None
        self.selected_result = "ideal"
        self.preview_motion = None
        self.preview_time = 0.
        self.preview_playing = False
        self.message = "Edit a release state, then Build action or Simulate release."
        self.last_poll = 0.
        self.loaded_completion = None
        self.library_index = 0
        self.history_index = 0
        self.scheme_paths = []
        self.history_paths = []
        self.capture_frames = 0
        self.initial_tab = args.tab
        self._load_spec(self.spec)
        self.refresh_library()
        if args.open_run:
            self.open_run(Path(args.open_run))
            if args.result_kind == "approach":
                self.select_result("approach")
            self.preview_time = min(args.preview_time, self.preview_duration())
        if hasattr(viewer, "register_ui_callback"):
            viewer.register_ui_callback(self.workbench_window, position="free")

    def _load_spec(self, spec):
        validate(spec)
        self.spec = copy.deepcopy(spec)
        t = self.spec["target"]
        self.target_position = np.array(t["p_world_m"], dtype=float)
        self.target_rpy_deg = np.array(t["rpy_xyz_deg"], dtype=float)
        self.target_linear_velocity = np.array(t["v_world_m_s"], dtype=float)
        self.target_angular_velocity = np.array(t["omega_world_rad_s"], dtype=float)
        self.show_release_state(use_initial_seed=True)

    def show_release_state(self, *, use_initial_seed=False):
        """Leave replay and solve from the previous release IK branch, not a replay frame."""
        self.preview_playing = False
        self.preview_motion = self.replay = None
        self.preview_time = 0.
        if self.full_result is not None:
            self.q = self.full_result.q_rad.copy()
        self.release_dirty = True
        self.release_solve_error = None
        try:
            self._solve_full(use_initial_seed=use_initial_seed)
            self.release_dirty = False
        except Exception as error:
            self.release_solve_error = str(error)

    def current_spec(self):
        spec = copy.deepcopy(self.spec)
        spec["target"] = {"p_world_m": self.target_position.tolist(), "rpy_xyz_deg": self.target_rpy_deg.tolist(),
                          "v_world_m_s": self.target_linear_velocity.tolist(), "omega_world_rad_s": self.target_angular_velocity.tolist()}
        return spec

    @property
    def stale(self):
        try:
            return self.run_hash is not None and self.run_hash != fingerprint(self.current_spec())
        except (ValueError, TypeError):
            return self.run_hash is not None

    def refresh_library(self):
        self.scheme_paths = sorted((ROOT / "schemes").glob("*.json"))
        self.history_paths = sorted((ROOT / "outputs").glob("*/design.json"), reverse=True)
        self.library_index = min(self.library_index, max(0, len(self.scheme_paths) - 1))
        self.history_index = min(self.history_index, max(0, len(self.history_paths) - 1))

    def start_job(self, task):
        if self.process is not None and self.process.poll() is None:
            self.message = "A job is already running. Cancel it before starting another."
            return
        try:
            spec = self.current_spec()
            directory = create_run(spec, task)
            self.run_dir, self.run_hash = directory, fingerprint(spec)
            self.result = self.audit_report = self.samples = self.replay = self.distribution = None
            self.preview_motion = None
            self.loaded_completion = None
            self.preview_playing = False
            self.process_log = (directory / "worker.log").open("w", encoding="utf-8")
            self.process = subprocess.Popen([sys.executable, "-s", str(ROOT / "worker.py"),
                                            "--run", str(directory), "--task", task],
                                           cwd=ROOT, stdout=self.process_log, stderr=subprocess.STDOUT)
            self.job = read_json(directory / "status.json")
            self.message = f"Started {task}. You can keep editing; this run uses a saved snapshot."
        except Exception as error:
            self.message = str(error)

    def cancel_job(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.job = {"state": "cancelled", "message": "Cancelled; partial files are not accepted results"}
            write_json(self.run_dir / "status.json", self.job)
            self.message = "Cancellation requested."

    def poll_job(self):
        if time.monotonic() - self.last_poll < .3:
            return
        self.last_poll = time.monotonic()
        if self.run_dir is None:
            return
        status_path = self.run_dir / "status.json"
        if status_path.exists():
            self.job = read_json(status_path)
        if self.process and self.process.poll() is not None:
            if self.process_log:
                self.process_log.close()
                self.process_log = None
            if self.job.get("state") in ("running", "queued"):
                self.job = {"state": "failed", "message": f"Worker exited ({self.process.returncode}); inspect worker.log"}
                write_json(status_path, self.job)
        if self.job.get("state") in ("complete", "failed", "cancelled") and self.loaded_completion != str(self.run_dir):
            self.loaded_completion = str(self.run_dir)
            self.read_result()
            self.refresh_library()

    def open_run(self, directory):
        if self.process and self.process.poll() is None:
            self.message = "Wait for or cancel the current job before opening another run."
            return
        self._load_spec(read_json(directory / "design.json"))
        self.run_dir = directory
        self.run_hash = fingerprint(self.current_spec())
        self.job = read_json(directory / "status.json")
        self.loaded_completion = str(directory)
        self.read_result()
        self.message = "Loaded the saved design and its recorded results."

    def read_result(self):
        self.result = self.audit_report = self.samples = self.replay = self.distribution = None
        self.preview_motion = None
        self.preview_playing = False
        if (self.run_dir / "release_audit.json").exists():
            self.audit_report = read_json(self.run_dir / "release_audit.json")
        if (self.run_dir / "release_samples.npz").exists():
            with np.load(self.run_dir / "release_samples.npz", allow_pickle=False) as data:
                self.samples = {k: data[k] for k in data.files}
        if (self.run_dir / "release.npz").exists():
            self.preview_motion = load_motion(self.run_dir / "release.npz")
        if (self.run_dir / "result.json").exists():
            self.result = read_json(self.run_dir / "result.json")
        self.select_result("ideal")
        # A background job must not replace the pose currently being edited.
        if self.stale:
            self.show_release_state()

    def select_result(self, name):
        self.selected_result = name
        self.replay = self.distribution = None
        self.preview_time = 0.
        self.preview_playing = False
        release_path = self.run_dir / "release.npz"
        self.preview_motion = load_motion(release_path) if release_path.exists() else None
        if not self.result or name not in self.result:
            return
        folder = self.run_dir / "ideal" if name == "ideal" else Path(self.result["approach"]["directory"])
        if (folder / "replay.npz").exists():
            with np.load(folder / "replay.npz", allow_pickle=False) as d:
                self.replay = {k: d[k] for k in ("times_s", "q", "v", "particle_positions", "retained_counts")}
            with np.load(folder / "distribution.npz", allow_pickle=False) as d:
                self.distribution = d["mass_grid"]
            self.preview_motion = load_motion(self.run_dir / "release.npz" if name == "ideal" else folder / "full.npz")

    def step(self):
        self.poll_job()
        self.sim_time += self.frame_dt
        if self.preview_playing:
            self.preview_time = min(self.preview_duration(), self.preview_time + self.frame_dt)
            if self.preview_time >= self.preview_duration():
                self.preview_playing = False
        if self.preview_motion is not None:
            t = min(self.preview_time, self.preview_motion.x[-1])
            self.q = self.preview_motion(t)
            self.qd = self.preview_motion(t, 1) if self.preview_time < self.preview_motion.x[-1] else np.zeros(6)
        self._apply_joint_pose()

    def preview_duration(self):
        if self.replay is not None:
            return float(self.replay["times_s"][-1])
        return float(self.preview_motion.x[-1]) if self.preview_motion is not None else 0.

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self._render_bucket_local_frame()
        p = self.target_position
        self.target_rotation = Rotation.from_euler("xyz", self.target_rpy_deg, degrees=True).as_matrix()
        for i, color in enumerate(((1., .2, .1), (.1, 1., .2), (.1, .4, 1.))):
            self._log_arrow(self.viewer, f"/designer/target_axis_{i}", p, .14 * self.target_rotation[:, i], color)
        actual_pose = self.kinematics.tool_pose_world(self.q, self.tool_reference_local)
        actual_p = actual_pose[:3, 3]
        # During replay use this frame's Jacobian and qdot, never an old IK twist.
        displayed_twist = self.kinematics.spatial_jacobian(self.q, self.tool_reference_local) @ self.qd
        for name, origin, vector, color, visible in (
            ("v", p, .3 * self.target_linear_velocity, (1., .85, .1), self.show_target_velocity),
            ("omega", p, .1 * self.target_angular_velocity, (.9, .2, 1.), self.show_target_velocity),
            ("solved_v", actual_p, .3 * displayed_twist[:3], (.1, 1., .3), self.show_solved_velocity),
            ("solved_omega", actual_p, .1 * displayed_twist[3:], (.1, .9, 1.), self.show_solved_velocity),
        ):
            if visible and np.linalg.norm(vector) > 1e-9:
                self._log_arrow(self.viewer, f"/designer/{name}", origin, vector, color)
            else:
                self.viewer.log_arrows(f"/designer/{name}", None, None, color, width=.009)
        self.viewer.log_points("/designer/solved_tool", wp.array([actual_p.astype(np.float32)], dtype=wp.vec3), .012,
                               wp.array([(.1, 1., .3)], dtype=wp.vec3))
        start = self.kinematics.tool_pose_world(self.initial_q, self.tool_reference_local)[:3, 3]
        self.viewer.log_points("/designer/fixed_start", wp.array([start.astype(np.float32)], dtype=wp.vec3), .02,
                               wp.array([(.2, .9, 1.)], dtype=wp.vec3))
        if self.preview_motion is not None:
            if not hasattr(self, "_path_key") or self._path_key != id(self.preview_motion):
                times = np.linspace(0, self.preview_motion.x[-1], 180)
                points = np.asarray([self.kinematics.tool_pose_world(q, self.tool_reference_local)[:3, 3]
                                     for q in self.preview_motion(times)], dtype=np.float32)
                self._path_points = (wp.array(points[:-1], dtype=wp.vec3), wp.array(points[1:], dtype=wp.vec3))
                self._path_key = id(self.preview_motion)
            self.viewer.log_lines("/designer/path", *self._path_points, (.95, .55, .15), width=.006)
        else:
            self.viewer.log_lines("/designer/path", None, None, None)
        if self.replay is not None:
            index = int(np.clip(np.searchsorted(self.replay["times_s"], self.preview_time, side="right") - 1,
                                0, len(self.replay["times_s"]) - 1))
            points = self.replay["particle_positions"][index]
            self.viewer.log_points("/designer/grains", wp.array(points, dtype=wp.vec3), .01,
                                   wp.array(np.tile([.95, .75, .25], (len(points), 1)).astype(np.float32), dtype=wp.vec3))
        else:
            self.viewer.log_points("/designer/grains", wp.empty(0, dtype=wp.vec3), .01)
        self.viewer.end_frame()
        self.capture_frames += 1
        if self.args.capture and self.capture_frames == 40:
            import pyglet
            pyglet.image.get_buffer_manager().get_color_buffer().save(self.args.capture)
        if self.args.test and self.capture_frames >= self.args.num_frames:
            self.viewer.close()

    @staticmethod
    def _plot(ui, label, values, height=65):
        ui.text(label)
        ui.plot_lines("##" + label, np.ascontiguousarray(values, dtype=np.float32),
                      graph_size=(max(100., ui.get_content_region_avail().x), height))

    def state_ui(self, ui):
        ui.text_wrapped("World-frame TOOL point [0.3, 0, 0.08] m, not flange position. RGB axes = target attitude.")
        live_changed, self.live_solve = ui.checkbox("Live solve while dragging", self.live_solve)
        ui.text_wrapped("Drag sliders; Ctrl+click a value for precise input.")
        changed = False
        maximum_v = float(self.full_cfg["maximum_linear_speed_m_s"])
        maximum_w = float(self.full_cfg["maximum_angular_speed_rad_s"])
        groups = (("Position (m)", self.target_position, ("p.x", "p.y", "p.z"),
                   ((-1.35, .10), (-1.70, .35), (-.25, .75)), "%.4f m"),
                  ("Fixed-axis XYZ RPY (deg)", self.target_rpy_deg, ("Roll", "Pitch", "Yaw"),
                   ((-180., 180.),) * 3, "%.2f deg"),
                  ("Linear velocity (m/s)", self.target_linear_velocity, ("v.x", "v.y", "v.z"),
                   ((-maximum_v, maximum_v),) * 3, "%.3f m/s"),
                  ("World angular velocity (rad/s)", self.target_angular_velocity, ("w.x", "w.y", "w.z"),
                   ((-maximum_w, maximum_w),) * 3, "%.3f rad/s"))
        for label, values, axes, ranges, fmt in groups:
            ui.separator_text(label)
            for i, axis in enumerate(axes):
                ui.set_next_item_width(max(160., ui.get_content_region_avail().x - 60.))
                edited, value = ui.slider_float(f"{axis}##{label}", float(values[i]), *ranges[i], fmt)
                if edited:
                    values[i] = value
                    changed = True
        if changed:
            self.release_dirty = True
            self.preview_playing = False
            self.preview_motion = None
            self.replay = None
        if self.live_solve and (changed or (live_changed and self.release_dirty)):
            self.show_release_state()
        if ui.button("Solve / show release state"):
            self.show_release_state()
        if ui.button("Use original fixed-start pose"):
            s = self.current_spec()
            s["target"].update({k: v for k, v in default_design()["target"].items() if k in ("p_world_m", "rpy_xyz_deg")})
            self._load_spec(s)
            self.preview_motion = self.replay = None
        ui.separator_text("Live release solution")
        if self.preview_motion is not None:
            ui.text_wrapped("Showing recorded replay. Drag a state slider or click Solve to return to the release pose.")
        if self.release_solve_error:
            ui.text_colored((1., .35, .2, 1.), "SOLVE FAILED")
            ui.text_wrapped(self.release_solve_error)
        elif self.release_dirty:
            ui.text_colored((1., .65, .1, 1.), "UNSOLVED EDIT: enable Live solve or click Solve")
        elif self.full_result:
            feasible = self.full_result.reachable and self.twist_reachable
            ui.text_colored((.2, 1., .4, 1.) if feasible else (1., .4, .2, 1.),
                            f"Pose: {'reachable' if self.full_result.reachable else 'UNREACHABLE'} | "
                            f"Twist: {'reachable' if self.twist_reachable else 'LIMITED'}")
            ui.text(f"Pose error: {self.full_result.position_error_m*1000:.2f} mm / {self.full_result.orientation_error_deg:.2f} deg")
            ui.text(f"Twist error: {self.twist_linear_error:.3f} m/s / {self.twist_angular_error:.3f} rad/s")
            ui.text(f"|v| target / solved: {np.linalg.norm(self.target_linear_velocity):.3f} / {np.linalg.norm(self.actual_twist[:3]):.3f} m/s")
            ui.text(f"|w| target / solved: {np.linalg.norm(self.target_angular_velocity):.3f} / {np.linalg.norm(self.actual_twist[3:]):.3f} rad/s")
        _, self.show_target_velocity = ui.checkbox("Target arrows: yellow v / magenta w", self.show_target_velocity)
        _, self.show_solved_velocity = ui.checkbox("Solved arrows: green v / cyan w", self.show_solved_velocity)
        ui.text_wrapped("Arrow length: v x 0.3 s, w x 0.1 m/(rad/s). Matching arrows overlap; zero speed has no arrow. During replay, solved arrows follow the recorded frame.")
        ui.text_wrapped("Targets are preserved, not clipped to feasible speeds. This is a pose/twist check; Build action checks the full motion.")
        ui.text_wrapped("Fixed-start approach uses the original migrated flange pose. Cyan point marks its tool point.")

    def action_ui(self, ui):
        a = self.spec["action"]
        ui.text_wrapped("Translation follows the INITIAL velocity direction. Distance is calculated. No forced endpoint or hidden retiming.")
        _, a["duration_s"] = ui.slider_float("Duration", a["duration_s"], .15, 4., "%.2f s")
        _, a["brake_start_fraction"] = ui.slider_float("Brake begins", a["brake_start_fraction"], 0., .8, "%.2f of T")
        shapes = ["early", "balanced", "late"]
        _, index = ui.combo("Brake shape", shapes.index(a["brake_shape"]), shapes)
        a["brake_shape"] = shapes[index]
        ui.separator_text("Flip (additional tool-local rotation)")
        axes = ["tool_x", "tool_y", "tool_z"]
        _, index = ui.combo("Axis", axes.index(a["flip_axis"]), axes)
        a["flip_axis"] = axes[index]
        _, a["flip_angle_deg"] = ui.slider_float("Angle", a["flip_angle_deg"], -120., 120., "%.1f deg")
        _, a["flip_start_fraction"] = ui.slider_float("Flip begins", a["flip_start_fraction"], 0., .9, "%.2f of T")
        _, a["flip_end_fraction"] = ui.slider_float("Flip ends", a["flip_end_fraction"], .1, 1., "%.2f of T")
        _, index = ui.combo("Flip shape", shapes.index(a["flip_shape"]), shapes)
        a["flip_shape"] = shapes[index]
        ui.text_wrapped("Initial world angular velocity decays smoothly through the action; the flip is added to this rotation. End linear/angular velocity = 0.")
        from motion import progress
        t = np.linspace(0, 1, 121)
        speed = 1 - progress(t, a["brake_start_fraction"], 1., a["brake_shape"])
        self._plot(ui, "Requested speed / initial speed", speed)
        if a["flip_end_fraction"] > a["flip_start_fraction"]:
            self._plot(ui, "Requested added flip (deg)", a["flip_angle_deg"] * progress(t, a["flip_start_fraction"], a["flip_end_fraction"], a["flip_shape"]))
        ui.text(f"Time axis: 0 to {a['duration_s']:.2f} seconds")
        ui.separator_text("Particles and approach")
        s = self.spec["simulation"]
        _, index = ui.combo("Newton substeps", [4, 8, 12].index(s["substeps"]), ["4 (screen)", "8 (verify)", "12"])
        s["substeps"] = [4, 8, 12][index]
        _, s["settle_seconds"] = ui.slider_float("Observe after stop", s["settle_seconds"], .5, 15., "%.1f s")
        _, index = ui.combo("Initial load", ["ideal", "velocity_jitter"].index(s["load_mode"]),
                            ["Rigid carried / zero relative speed", "Rigid carried + relative velocity noise"])
        s["load_mode"] = ["ideal", "velocity_jitter"][index]
        if index:
            _, s["velocity_noise_m_s"] = ui.slider_float("Noise std", s["velocity_noise_m_s"], 0., .2, "%.3f m/s")
            _, s["seed"] = ui.input_int("Seed", int(s["seed"]))
        pre = self.spec["approach"]
        _, pre["allow_pullback"] = ui.checkbox("Automatically search pullback preparations", pre["allow_pullback"])
        _, pre["screen_candidates"] = ui.slider_int("Screen candidates", pre["screen_candidates"], 1, 6)
        _, pre["minimum_retention"] = ui.slider_float("Retention target", pre["minimum_retention"], .5, 1., "%.3f")

    def heatmap(self, ui, grid):
        ui.text("Settled mass / initial total mass (bucket XY)")
        pos = ui.get_cursor_screen_pos()
        size = min(280., ui.get_content_region_avail().x)
        cell = size / grid.shape[0]
        draw = ui.get_window_draw_list()
        maximum = max(float(grid.max()), 1e-9)
        for ix in range(grid.shape[0]):
            for iy in range(grid.shape[1]):
                value = np.sqrt(float(grid[ix, iy]) / maximum)
                r, g, b = int(22 + 233 * value), int(30 + 145 * value), int(48 - 25 * value)
                color = r | g << 8 | b << 16 | 255 << 24
                # np.histogram2d stores grid[x,y]; show +Y upward.
                draw.add_rect_filled((pos.x + ix * cell, pos.y + (grid.shape[1] - 1 - iy) * cell),
                                     (pos.x + (ix + 1) * cell, pos.y + (grid.shape[1] - iy) * cell), color)
        ui.dummy((size, size))
        ui.text(f"+X right, +Y up | max cell {maximum:.2%}")

    def results_ui(self, ui):
        if self.run_dir is None:
            ui.text_wrapped("Build or simulate an action to see results here.")
            return
        if self.preview_motion is None and (self.run_dir / "release.npz").exists():
            if ui.button("Show saved replay"):
                self.select_result(self.selected_result)
        if self.result and "ideal" in self.result:
            names = ["ideal"] + (["approach"] if "approach" in self.result else [])
            labels = ["Ideal release only", "Fixed-start full motion"][:len(names)]
            changed, index = ui.combo("Result", names.index(self.selected_result) if self.selected_result in names else 0, labels)
            if changed:
                self.select_result(names[index])
        if self.preview_motion is not None:
            if ui.button("Pause preview" if self.preview_playing else "Play preview"):
                if self.preview_time >= self.preview_duration():
                    self.preview_time = 0.
                self.preview_playing = not self.preview_playing
            ui.same_line()
            if ui.button("Restart preview"):
                self.preview_time, self.preview_playing = 0., True
            changed, self.preview_time = ui.slider_float("Replay time", self.preview_time, 0., self.preview_duration(), "%.3f s")
            if changed:
                self.preview_playing = False
        if self.audit_report:
            report = self.audit_report
            ui.text(f"Release mechanical audit: {'PASS' if report['valid'] else 'FAIL'}")
            p = report["polynomial"]
            ui.text(f"Joint utilization: v {p['velocity_ratio']:.1%}, a {p['acceleration_ratio']:.1%}, jerk {p['jerk_ratio']:.1%}")
            ui.text(f"Follow distance: {report['follow_distance_m']:.3f} m")
            ui.text(f"Max action tracking: {report['action_position_error_max_m']*1000:.2f} mm / {report['action_orientation_error_max_deg']:.3f} deg")
        if self.samples is not None:
            self._plot(ui, "Achieved release speed (m/s)", np.linalg.norm(self.samples["twist"][:, :3], axis=1))
            self._plot(ui, "Achieved angular speed (rad/s)", np.linalg.norm(self.samples["twist"][:, 3:], axis=1))
        if self.result and self.selected_result in self.result:
            r = self.result[self.selected_result]
            if self.selected_result == "ideal":
                ui.text_wrapped("IDEAL INITIALIZATION: particles are placed in the release pose with a rigid velocity field. This does not certify approach feasibility.")
            else:
                ui.text_wrapped("CONTINUOUS FROM FIXED START: material is not reset at release.")
                ui.text(f"Retention gate: {'PASS' if r['retention_pass'] else 'FAIL'}")
            ui.text(f"Min pre-release retention: {r['minimum_pre_release_retention']:.2%}")
            ui.text(f"Release retention: {r['release_retained_fraction']:.2%}")
            rms = r["release_relative_speed_rms_m_s"]
            ui.text(f"Release relative-speed RMS: {rms:.3f} m/s" if rms is not None else "No retained grains at release")
            m = r["metrics"]
            ui.text(f"Settled usable: {m['settled_usable_fraction']:.2%}")
            ui.text(f"Outside pot (includes carried): {m['outside_bucket_fraction']:.2%}")
            ui.text(f"Shovel residual (AABB): {m['shovel_residual_fraction']:.2%}")
            ui.text_wrapped("Outside-pot and shovel-residual categories overlap; they are not added together.")
            ui.text(f"Settling criterion: {'stable' if m['stable_observation'] else 'NOT STABLE - extend observation'}")
            if self.replay is not None:
                self._plot(ui, "On-shovel fraction (instantaneous)", self.replay["retained_counts"] / r["particle_count"])
                ui.text(f"Whole replay: 0 to {self.preview_duration():.2f}s; release at {r['release_time_s']:.3f}s")
            if self.distribution is not None:
                self.heatmap(ui, self.distribution)
            if "comparison" in self.result:
                ui.text(f"Ideal vs full mass TV: {self.result['comparison']['mass_grid_tv_including_missing_mass']:.3f}")
        ui.text_wrapped(str(self.run_dir))

    def library_ui(self, ui):
        _, self.spec["name"] = ui.input_text("Scheme name", self.spec["name"])
        if ui.button("Save scheme as a new version"):
            try:
                path = save_design(self.current_spec())
                self.message = f"Saved {path.name}"
                self.refresh_library()
            except Exception as e:
                self.message = str(e)
        if ui.button("Refresh library / history"):
            self.refresh_library()
        if self.scheme_paths:
            _, self.library_index = ui.combo("Saved schemes", self.library_index, [p.stem for p in self.scheme_paths])
            if ui.button("Load selected scheme"):
                try:
                    self._load_spec(read_json(self.scheme_paths[self.library_index]))
                    self.preview_motion = self.replay = None
                    self.message = "Loaded scheme. Previous results are historical until rebuilt."
                except Exception as e:
                    self.message = str(e)
        if self.history_paths:
            _, self.history_index = ui.combo("Run history", self.history_index, [p.parent.name for p in self.history_paths])
            if ui.button("Open run and its design snapshot"):
                try:
                    self.open_run(self.history_paths[self.history_index].parent)
                except Exception as e:
                    self.message = str(e)
        if ui.collapsing_header("Fixed start reference"):
            start = fixed_start()
            ui.text_wrapped("Base-frame FLANGE pose; same as the migrated original collection:")
            ui.text_wrapped(str(start["flange_pose_base"]))
            ui.text_wrapped("Joint degrees: " + str(np.rad2deg(start["q_rad"]).round(5).tolist()))

    def gui(self, ui):
        ui.text_wrapped("Release Designer controls are in the floating workbench window on the right.")

    def workbench_window(self, ui):
        display = ui.get_io().display_size
        ui.set_next_window_pos((max(320., display.x - 545.), 10.), ui.Cond_.first_use_ever)
        ui.set_next_window_size((530., max(500., display.y - 20.)), ui.Cond_.first_use_ever)
        visible, _ = ui.begin("Release Workbench", flags=ui.WindowFlags_.no_collapse)
        if visible:
            self.workbench_content(ui)
        ui.end()

    def workbench_content(self, ui):
        ui.text("EC66 RELEASE DESIGNER | V1")
        if ui.small_button("Top view") and hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(.12, -.96, 2.8), -89., -90.)
        ui.same_line()
        if ui.small_button("Oblique view") and hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(-2.5, 1.4, 1.5), -28., -49.)
        ui.text_wrapped(self.message)
        busy = self.process is not None and self.process.poll() is None
        if self.job:
            ui.text_wrapped(f"{self.job.get('state', '').upper()}: {self.job.get('message', '')}")
        if self.stale:
            ui.text_colored((1., .65, .1, 1.), "EDITED: displayed results belong to an older snapshot")
        ui.begin_disabled(busy)
        if ui.button("1 Build action"):
            self.start_job("build")
        ui.same_line()
        if ui.button("2 Simulate release"):
            self.start_job("simulate")
        if ui.button("3 Plan approach + compare full motion"):
            self.start_job("approach")
        ui.end_disabled()
        if busy and ui.button("Cancel running job"):
            self.cancel_job()
        ui.separator()
        if ui.begin_tab_bar("Designer tabs"):
            for label, method in (("State", self.state_ui), ("Action", self.action_ui),
                                  ("Results", self.results_ui), ("Library", self.library_ui)):
                flags = ui.TabItemFlags_.set_selected if self.initial_tab == label else 0
                visible, _ = ui.begin_tab_item(label, flags=flags)
                if visible:
                    method(ui)
                    ui.end_tab_item()
            ui.end_tab_bar()
            self.initial_tab = None

    def test_final(self):
        # Parent test_final writes old-project emergency-brake artifacts; never call it.
        assert np.isfinite(self.state.body_q.numpy()).all()
        np.testing.assert_allclose(self.initial_q, fixed_start()["q_rad"], atol=1e-10)
        if self.process and self.process.poll() is None:
            self.cancel_job()


def main():
    parser = FullStateExample.create_parser()
    parser.add_argument("--scheme", type=Path)
    parser.add_argument("--open-run", type=Path)
    parser.add_argument("--capture")
    parser.add_argument("--tab", choices=("State", "Action", "Results", "Library"), default="State")
    parser.add_argument("--result-kind", choices=("ideal", "approach"), default="ideal")
    parser.add_argument("--preview-time", type=float, default=0.)
    parser.set_defaults(show_bucket_frame=True)
    viewer, args = newton.examples.init(parser)
    example = Designer(viewer, args)
    try:
        newton.examples.run(example, args)
    finally:
        if example.process and example.process.poll() is None:
            example.cancel_job()
        if example.process_log:
            example.process_log.close()


if __name__ == "__main__":
    main()
