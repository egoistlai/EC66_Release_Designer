"""Exercise slider edits against real EC66 IK/FK without an OpenGL window."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gui import Designer, FullStateExample, newton
from design import default_design


def controls(edits=None, live=None):
    ui = MagicMock()
    edits = edits or {}
    ui.get_content_region_avail.return_value = SimpleNamespace(x=480.)
    ui.slider_float.side_effect = lambda label, value, *args: (
        label in edits, edits.get(label, value))
    ui.checkbox.side_effect = lambda label, value: (
        (live is not None and live != value, live)
        if label == "Live solve while dragging" and live is not None else (False, value))
    ui.button.return_value = False
    return ui


class LiveEditorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(sys, "argv", ["editor-test", "--viewer", "null"]):
            viewer, args = newton.examples.init(FullStateExample.create_parser())
        for key, value in dict(scheme=None, tab="State", open_run=None, capture=None).items():
            setattr(args, key, value)
        cls.editor = Designer(viewer, args)

    def setUp(self):
        self.editor._load_spec(default_design())
        self.editor.live_solve = True

    def test_slider_changes_solve_pose_and_twist_before_next_step(self):
        e = self.editor
        old_q = e.q.copy()
        e.state_ui(controls({"p.z##Position (m)": float(e.target_position[2] + .025),
                             "Yaw##Fixed-axis XYZ RPY (deg)": float(e.target_rpy_deg[2] + 3.),
                             "v.y##Linear velocity (m/s)": -.3,
                             "w.z##World angular velocity (rad/s)": .15}))
        self.assertTrue(e.full_result.reachable)
        self.assertFalse(e.release_dirty)
        self.assertGreater(np.linalg.norm(e.q - old_q), .01)
        actual_pose = e.kinematics.tool_pose_world(e.q, e.tool_reference_local)
        np.testing.assert_allclose(actual_pose[:3, 3], e.target_position, atol=1e-4)
        np.testing.assert_allclose(e.actual_twist,
                                   e.kinematics.spatial_jacobian(e.q, e.tool_reference_local) @ e.qd)
        solved = e.q.copy()
        e.step()
        np.testing.assert_allclose(e.q, solved)

    def test_edit_leaves_replay_and_uses_previous_release_seed(self):
        e = self.editor
        release_q = e.q.copy()
        e.q = release_q + .2  # Simulate a frame later in recorded playback.
        e.preview_motion = object()
        e.replay = {"old": True}
        e.preview_playing = True
        seeds = []
        original = e._solve_full

        def record_seed(**kwargs):
            seeds.append(e.q.copy())
            original(**kwargs)

        with patch.object(e, "_solve_full", side_effect=record_seed):
            e.state_ui(controls({"v.x##Linear velocity (m/s)": -.2}))
        np.testing.assert_allclose(seeds[0], release_q)
        self.assertIsNone(e.preview_motion)
        self.assertIsNone(e.replay)
        self.assertFalse(e.preview_playing)
        solved = e.q.copy()
        e.step()
        np.testing.assert_allclose(e.q, solved)

    def test_reenabling_live_solve_applies_pending_edits(self):
        e = self.editor
        old_q = e.q.copy()
        e.state_ui(controls({"p.z##Position (m)": float(e.target_position[2] + .025)}, live=False))
        self.assertTrue(e.release_dirty)
        np.testing.assert_allclose(e.q, old_q)
        e.state_ui(controls(live=True))
        self.assertFalse(e.release_dirty)
        self.assertGreater(np.linalg.norm(e.q - old_q), .01)

    def test_velocity_arrows_follow_joint_state_and_clear_at_zero(self):
        e = self.editor
        e.target_angular_velocity[2] = .2
        e.show_release_state()
        # Mimic replay velocities different from the stored release solution.
        e.qd *= -.5
        origin = e.kinematics.tool_pose_world(e.q, e.tool_reference_local)[:3, 3]
        actual = e.kinematics.spatial_jacobian(e.q, e.tool_reference_local) @ e.qd
        with patch.object(e.viewer, "log_arrows") as arrows:
            e.render()
            logged = {call.args[0]: call.args for call in arrows.call_args_list}
            for name, vector in (("solved_v", .3 * actual[:3]), ("solved_omega", .1 * actual[3:])):
                args = logged["/designer/" + name]
                np.testing.assert_allclose(args[1].numpy()[0], origin, atol=1e-6)
                np.testing.assert_allclose(args[2].numpy()[0] - args[1].numpy()[0], vector, atol=1e-6)
            e.qd.fill(0.)
            arrows.reset_mock()
            e.render()
            logged = {call.args[0]: call.args for call in arrows.call_args_list}
            self.assertIsNone(logged["/designer/solved_v"][1])
            self.assertIsNone(logged["/designer/solved_omega"][1])


if __name__ == "__main__":
    unittest.main()
