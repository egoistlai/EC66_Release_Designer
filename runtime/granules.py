"""Newton XPBD sphere grains: explicit particle-particle contacts prevent collapse."""
import numpy as np
import newton,warp as wp
from scipy.spatial.transform import Rotation as R
class Granules:
 def __init__(self,scene,points):
  b=newton.ModelBuilder();body=b.add_body(mass=0,is_kinematic=True,label='prescribed_shovel')
  cfg=newton.ModelBuilder.ShapeConfig(density=0,mu=0.8,has_shape_collision=False)
  mount=scene.mount
  b.add_shape_mesh(body,xform=wp.transform(wp.vec3(*mount[:3,3]),wp.quat(*R.from_matrix(mount[:3,:3]).as_quat())),mesh=newton.Mesh(vertices=np.asarray(scene.shovel_mesh.vertices,np.float32),indices=np.asarray(scene.shovel_mesh.faces,np.int32).ravel()),cfg=cfg)
  b.add_shape_mesh(-1,mesh=newton.Mesh(vertices=np.asarray(scene.bucket_mesh.vertices,np.float32),indices=np.asarray(scene.bucket_mesh.faces,np.int32).ravel()),cfg=cfg)
  z=scene.cfg['bucket_xyz'][2];b.add_shape_box(-1,xform=wp.transform(wp.vec3(0,0,z-.25),wp.quat_identity()),hx=5,hy=5,hz=.25,cfg=cfg)
  d=scene.cfg['particle_diameter_m'];mass=scene.cfg['density']*np.pi/6*d**3
  for p in points:b.add_particle(pos=wp.vec3(*p),vel=wp.vec3(0),mass=mass,radius=d/2)
  self.model=b.finalize();self.model.particle_mu=0.8;self.model.particle_cohesion=0.0
  self.solver=newton.solvers.SolverXPBD(self.model,iterations=20,soft_contact_relaxation=.25)
  self.a=self.model.state();self.b=self.model.state();self.scene=scene
 def reset(self,points):
  self.a=self.model.state();self.b=self.model.state();self.a.particle_q.assign(np.asarray(points,np.float32));self.sync()
 def sync(self):
  pose=self.pose_override if hasattr(self,'pose_override') else self.scene.state.body_q.numpy()[self.scene.flange_id];self.a.body_q.assign(np.asarray([pose],np.float32));self.a.body_qd.zero_()
  if not hasattr(self,"pose_override"):self.a.body_qd.assign(self.scene.state.body_qd.numpy()[self.scene.flange_id:self.scene.flange_id+1])
 def step(self,dt):
  self.sync();self.a.clear_forces();contacts=self.model.collide(self.a);self.solver.step(self.a,self.b,control=None,contacts=contacts,dt=dt);self.a,self.b=self.b,self.a
