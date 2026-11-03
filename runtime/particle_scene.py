"""Robot/particle scene base; only the requested trajectory is loaded."""
from pathlib import Path
import argparse
import json
import time
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
import newton
import warp as wp
from my_scene import Example as Scene
from granules import Granules
ROOT = Path(__file__).resolve().parent

class Arm:

    def __init__(self):
        root = ET.parse(ROOT / 'assets/ec66_description.urdf').getroot()
        self.chain = []
        self.low = []
        self.high = []
        for j in root.findall('joint'):
            o = j.find('origin')
            axis = j.find('axis')
            moving = j.get('type') == 'revolute'
            self.chain.append((np.fromstring(o.get('xyz'), sep=' '), R.from_euler('xyz', np.fromstring(o.get('rpy'), sep=' ')).as_matrix(), np.fromstring(axis.get('xyz'), sep=' ') if moving else None))
            if moving:
                self.low.append(float(j.find('limit').get('lower')))
                self.high.append(float(j.find('limit').get('upper')))
        self.low = np.array(self.low)
        self.high = np.array(self.high)

    def fk(self, q):
        p = np.zeros(3)
        r = np.eye(3)
        i = 0
        for offset, origin, axis in self.chain:
            p = p + r @ offset
            r = r @ origin
            if axis is not None:
                r = r @ R.from_rotvec(axis * q[i]).as_matrix()
                i += 1
        return (p, r)

    def residual(self, q, p, r):
        a, b = self.fk(q)
        return np.r_[a - p, 0.25 * R.from_matrix(r @ b.T).as_rotvec()]

    def errors(self, q, p, r):
        a, b = self.fk(q)
        return (np.linalg.norm(a - p) * 1000, np.degrees(R.from_matrix(r @ b.T).magnitude()))

class Example:
    def __init__(self, viewer, args):
        self.viewer, self.args = viewer, args
        self.arm = Arm()
        self.playing = args.autoplay
        self.sim_time = 0.
        self.last = time.monotonic()
        self.frames = 0
        config = ROOT / 'scene_config_particles.json'
        sa = argparse.Namespace(config=str(config), static=True, scene_only=True,
                                results=str(ROOT.parent / 'outputs/scene'))
        self.scene = Scene(viewer, sa)
        self.W = self.scene.world_from_base
        self.grains = Granules(self.scene, self.scene.initial_particles)
        self.files = [Path(args.trajectory).resolve()]
        self.load(0)
        if hasattr(viewer, 'set_camera'):
            viewer.set_camera(wp.vec3(-2.5, 1.4, 1.5), -28, -49)

    def load(self, i):
        self.index = i
        d = np.load(self.files[i])
        self.qs = d['q']
        self.poses = d['poses']
        self.duration = (len(self.qs) - 1) * 0.1
        self.sim_time = 0.0
        self.last = time.monotonic()
        self.slerp = Slerp(np.arange(len(self.qs)) * 0.1, R.from_quat(self.poses[:, 3:7]))
        self.update()
        self.reset_particles()

    def reset_particles(self):
        scene = self.scene
        p, r = self.arm.fk(self.qs[0])
        flange = np.eye(4)
        flange[:3, :3] = r
        flange[:3, 3] = p
        shovel = self.W @ flange @ scene.mount
        points = scene.fill_local @ shovel[:3, :3].T + shovel[:3, 3]
        if self.grains is not None:
            self.grains.reset(points)
            scene.sand_state = self.grains.a
            scene.sand_model = self.grains.model
            self.physics_steps = 0
            return
        scene.sand_state = scene.sand_model.state()
        scene.sand_state.particle_q.assign(points.astype(np.float32))
        scene.sand_state.particle_qd.zero_()
        scene.sand_state.body_q = wp.clone(scene.state.body_q)
        scene.sand_state.body_qd = wp.clone(scene.state.body_qd)
        scene.sand_state.body_f = wp.zeros_like(scene.state.body_f)
        self.physics_steps = 0

    def update(self):
        f = self.sim_time / 0.1
        i = min(int(f), len(self.qs) - 2)
        u = f - i
        self.q = (1 - u) * self.qs[i] + u * self.qs[i + 1]
        p = (1 - u) * self.poses[i, :3] + u * self.poses[i + 1, :3]
        r = self.slerp([self.sim_time]).as_matrix()[0]
        self.error = self.arm.errors(self.q, p, r)
        self.scene.q_host[self.scene.joint_q_ids] = self.q
        self.scene.qd_host.fill(0)
        if self.playing:
            self.scene.qd_host[self.scene.joint_dof_ids] = (self.qs[i + 1] - self.qs[i]) / 0.1
        self.scene.state.joint_q.assign(self.scene.q_host)
        self.scene.state.joint_qd.assign(self.scene.qd_host)
        newton.eval_fk(self.scene.model, self.scene.state.joint_q, self.scene.state.joint_qd, self.scene.state)
        self.target = self.W[:3, :3] @ p + self.W[:3, 3]
        self.current_segment = i
