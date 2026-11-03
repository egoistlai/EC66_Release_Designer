"""Numerical contracts that protect release intent and physical initialization."""
from pathlib import Path
import copy
import sys
import tempfile
import unittest

import numpy as np
from scipy.interpolate import PPoly
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from design import default_design, validate, fixed_start, fingerprint
from motion import (carried_time, progress, desired_pose, spline_motion, polynomial_limits,
                    build_release, load_motion)


class TimingTests(unittest.TestCase):
    def test_brake_integral_matches_velocity(self):
        for shape in ("early", "balanced", "late"):
            t = np.linspace(.001, 1.199, 2000)
            eps = 1e-5
            derivative = (carried_time(t + eps, 1.2, .2, shape) - carried_time(t - eps, 1.2, .2, shape)) / (2 * eps)
            expected = 1 - progress(t, .24, 1.2, shape)
            np.testing.assert_allclose(derivative, expected, atol=1e-8)
            self.assertTrue(np.all(np.diff(expected) <= 1e-12))

    def test_timing_shape_changes_distance(self):
        lengths = [float(carried_time(1., 1., 0., shape)) for shape in ("early", "balanced", "late")]
        self.assertLess(lengths[0], lengths[1])
        self.assertLess(lengths[1], lengths[2])

    def test_nonzero_initial_angular_velocity_is_not_reset(self):
        s = default_design()
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler("xyz", [.3, -.4, .2]).as_matrix()
        twist = np.array([.2, -.1, .1, .3, -.2, .4])
        eps = 1e-6
        p = desired_pose(s, pose, twist, [0., eps])
        np.testing.assert_allclose((p[1, :3, 3] - p[0, :3, 3]) / eps, twist[:3], atol=1e-7)
        angular = Rotation.from_matrix(p[1, :3, :3] @ p[0, :3, :3].T).as_rotvec() / eps
        np.testing.assert_allclose(angular, twist[3:], atol=1e-7)

    def test_tool_local_flip_and_terminal_pose(self):
        s = default_design()
        s['action']['flip_angle_deg'] = 30.
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler("z", .7).as_matrix()
        p = desired_pose(s, pose, np.zeros(6), [s['action']['duration_s']])[0]
        expected = pose[:3, :3] @ Rotation.from_euler("y", 30, degrees=True).as_matrix()
        np.testing.assert_allclose(p[:3, :3], expected, atol=1e-12)

    def test_invalid_windows_and_nonfinite_input_fail(self):
        s = default_design()
        s['action']['flip_start_fraction'] = .9
        s['action']['flip_end_fraction'] = .2
        with self.assertRaises(ValueError):
            validate(s)
        s = default_design()
        s['target']['v_world_m_s'][0] = float('nan')
        with self.assertRaises(ValueError):
            validate(s)

    def test_fingerprint_tracks_intent_not_display_name(self):
        s = default_design()
        same = copy.deepcopy(s)
        same['name'] = 'Renamed'
        self.assertEqual(fingerprint(s), fingerprint(same))
        same['action']['flip_angle_deg'] += 1
        self.assertNotEqual(fingerprint(s), fingerprint(same))


class PolynomialTests(unittest.TestCase):
    @staticmethod
    def check(motion, low=-10., high=10., jerk=1000.):
        return polynomial_limits(motion, np.full(6, low), np.full(6, high),
                                 np.full(6, 100.), np.full(6, 100.), np.full(6, jerk))

    def test_interior_extremum_cannot_hide_between_endpoints(self):
        # q(t)=t-t²: endpoints are zero, internal maximum is .25.
        coefficients = np.tile(np.array([-1., 1., 0.])[:, None, None], (1, 1, 6))
        result = self.check(PPoly(coefficients, [0, 1]), low=-1., high=.2)
        self.assertFalse(result['position_valid'])
        self.assertFalse(result['valid'])

    def test_acceleration_jump_rejected_even_with_small_piecewise_jerk(self):
        c = np.zeros((3, 2, 6))
        c[0, 1] = 1.
        result = self.check(PPoly(c, [0., 1., 2.]))
        self.assertEqual(result['jerk_ratio'], 0.)
        self.assertGreater(result['join_jumps_q_qd_qdd'][2], 1.)
        self.assertFalse(result['valid'])

    def test_absolute_jerk_limit(self):
        c = np.zeros((4, 1, 6))
        c[0] = 1.
        result = self.check(PPoly(c, [0, 1]), jerk=5.)
        self.assertAlmostEqual(result['jerk_ratio'], 1.2)
        self.assertFalse(result['valid'])

    def test_via_point_is_not_forced_to_stop(self):
        q = np.tile(np.array([0., .2, 1.])[:, None], (1, 6))
        m = spline_motion(np.array([0., .4, 1.]), q, np.zeros(6), np.zeros(6))
        self.assertGreater(np.linalg.norm(m(.4, 1)), .1)
        for t in (0., 1.):
            np.testing.assert_allclose(m(t, 1), 0, atol=1e-10)
            np.testing.assert_allclose(m(t, 2), 0, atol=1e-10)
        self.assertLess(max(self.check(m)['join_jumps_q_qd_qdd']), 1e-8)


class ReleaseIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.spec = default_design()
        cls.context, cls.motion, cls.report = build_release(cls.spec, cls.temp.name)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_migrated_fixed_start_and_valid_default(self):
        np.testing.assert_allclose(self.context.q_start, fixed_start()['q_rad'], atol=1e-12)
        expected = fixed_start()['flange_pose_base']
        flange = self.context.kinematics.flange_pose_base(self.context.q_start)
        np.testing.assert_allclose(flange[:3, 3], expected['p_base_m'], atol=1e-6)
        self.assertTrue(self.report['valid'])

    def test_release_state_and_stationary_stop(self):
        motion, context = self.motion, self.context
        twist = context.kinematics.tool_twist_world(motion(0), motion(0, 1), context.tool_reference)
        target = self.spec['target']
        np.testing.assert_allclose(twist[:3], target['v_world_m_s'], atol=.01)
        np.testing.assert_allclose(twist[3:], target['omega_world_rad_s'], atol=.03)
        np.testing.assert_allclose(motion(motion.x[-1], 1), 0, atol=1e-9)
        np.testing.assert_allclose(motion(motion.x[-1], 2), 0, atol=1e-8)

    def test_saved_release_event_is_start_not_brake_end(self):
        path = Path(self.temp.name) / 'release.npz'
        with np.load(path, allow_pickle=False) as d:
            self.assertEqual(float(d['release_state_s']), 0.)
            self.assertEqual(float(d['release_action_start_s']), 0.)
        restored = load_motion(path)
        for order in range(4):
            np.testing.assert_allclose(restored(np.linspace(0, 1, 71), order), self.motion(np.linspace(0, 1, 71), order))


if __name__ == '__main__':
    unittest.main()
