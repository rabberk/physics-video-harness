"""Adapt Newton 1.7 official coupling example, retaining its impulse exchange."""
import argparse,json,time
from pathlib import Path
import numpy as np
import warp as wp
import newton
from vendor.newton_twoway import Example as Base
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
class Viewer:
 def set_model(self,m):pass
 def apply_forces(self,s):pass
class Example(Base):
 def capture(self):self.graph=None
 def _emit_rigid_bodies(self,b):
  ball=b.add_body(xform=wp.transform(wp.vec3(-.65,0,.121),wp.quat_identity()))
  b.add_shape_sphere(ball,radius=.12,cfg=newton.ModelBuilder.ShapeConfig(density=1500,mu=.6))
 def _emit_particles(self,b,dx):
  rng=np.random.default_rng(0);step=.025
  pts=np.array([(x,y,z) for x in np.arange(-.15,.36,step) for y in np.arange(-.25,.26,step) for z in np.arange(.02,.26,step) if z<.26-abs(x-.1)*.5-abs(y)*.5]);pts+=rng.uniform(-.004,.004,pts.shape)
  for q in pts:b.add_particle(pos=wp.vec3(q),vel=wp.vec3(0),mass=1500*step**3,radius=step/2,custom_attributes={'mpm:friction':.65})
wp.init()
with wp.ScopedDevice('cpu'):
 start=time.time();e=Example(Viewer(),None);print('initialized',flush=True)
 # Settling uses actual coupled solve with sphere at rest on the floor.
 for i in range(60):
  e.step()
  if i%10==0:print('settle',i,flush=True)
 e.state_0.body_qd.assign(np.array([[1.2,0,0,0,10,0]],dtype=np.float32))
 previous=e.sand_model.state()
 grains=e.mpm_solver.sample_render_grains(e.sand_state_0,64)
 pos=[];g=[];b=[]
 for frame in range(73):
  pos.append(e.sand_state_0.particle_q.numpy().copy());g.append(grains.numpy().reshape(-1,3).copy());b.append(e.state_0.body_q.numpy()[0].copy())
  if frame%12==0:print('frame',frame,'ball',b[-1][:3],flush=True)
  for k in range(4):
   previous.particle_q.assign(e.sand_state_0.particle_q);previous.particle_qd.assign(e.sand_state_0.particle_qd);previous.mpm.particle_transform.assign(e.sand_state_0.mpm.particle_transform)
   e.step()
   e.mpm_solver.update_particle_frames(previous,e.sand_state_0,e.frame_dt)
   e.mpm_solver.update_render_grains(previous,e.sand_state_0,grains,e.frame_dt)
 q=np.asarray(pos);gs=np.asarray(g);bs=np.asarray(b)
 np.savez_compressed(out/'states.npz',positions=gs,simulation_positions=q,ball_positions=bs[:,:3],ball_quaternions=bs[:,3:][:,[3,0,1,2]],t=np.arange(73)/25)
 checks={'finite':bool(np.isfinite(gs).all() and np.isfinite(bs).all()),'minimum_particle_z_m':float(q[:,:,2].min()),'ball_x_range_m':[float(bs[:,0].min()),float(bs[:,0].max())]};checks['passed']=bool(checks['finite'] and checks['minimum_particle_z_m']>-.06)
 meta={'case':'newton_rolling_sand','backend':'Newton 1.7 Implicit MPM + XPBD, official impulse coupling','geometry':'particles','radius':.003,'tank':False,'fps':25,'frames':73,'dynamic_sphere':{'radius':.12},'checks':checks,'simulation_particles':q.shape[1],'visual_grains':gs.shape[1],'elapsed_seconds':time.time()-start}
 (out/'simulation.json').write_text(json.dumps(meta,indent=2));print(meta,flush=True)
