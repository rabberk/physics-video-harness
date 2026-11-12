"""Newton XPBD + implicit MPM scene, retaining official impulse exchange.
The orchestration and kernels follow NVIDIA's Apache-2.0 two-way example.
"""
import argparse,json,time,sys,math
from pathlib import Path
import numpy as np
import warp as wp
import newton
from newton.solvers import SolverImplicitMPM,SolverXPBD
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vendor.newton_twoway import compute_body_forces,subtract_body_force

def height(x):return 1.64-.2*x
# Last 0.8 m becomes a smooth upward launch lip; upstream track is unchanged.
def launch_height(x):
 u=max(0.,x-.85);return height(x)+.55*u*u

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--frames',type=int,default=124);p.add_argument('--settle',type=int,default=24);p.add_argument('--sand-friction',type=float,default=.8);p.add_argument('--sand-young',type=float,default=1.e15);p.add_argument('--voxel',type=float,default=.025);p.add_argument('--mpm-substeps',type=int,default=5);p.add_argument('--grid-type',choices=['fixed','sparse','dense'],default='fixed');p.add_argument('--sand-start',type=float,default=.30);p.add_argument('--sand-end',type=float,default=.79);p.add_argument('--sand-thickness',type=float,default=.105);p.add_argument('--sand-half-width',type=float,default=.24);p.add_argument('--particle-spacing',type=float,default=.0125);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
 wp.init();started=time.time()
 with wp.ScopedDevice(a.device):
  builder=newton.ModelBuilder();cfg=newton.ModelBuilder.ShapeConfig(density=500,mu=.5,restitution=.02)
  ball=builder.add_body(xform=wp.transform(wp.vec3(-3.12,0,height(-3.12)+.185),wp.quat_identity()))
  builder.add_shape_sphere(ball,radius=.18,cfg=cfg)
  static=newton.ModelBuilder.ShapeConfig(density=0,mu=.7,restitution=.01)
  def box(center,size,angle=0,cfg_override=None):
   builder.add_shape_box(body=-1,xform=wp.transform(wp.vec3(*center),wp.quat_from_axis_angle(wp.vec3(0,1,0),float(angle))),hx=size[0]/2,hy=size[1]/2,hz=size[2]/2,cfg=cfg_override or static)
  angle=math.atan(.2)
  box((-1.325,0,height(-1.325)-.06),(4.45,.86,.12),angle)
  for y in [-.46,.46]:box((-1.325,y,height(-1.325)+.055),(4.45,.065,.22),angle)
  for left in np.linspace(.85,1.62,23):
   right=left+.035;mid=(left+right)/2;z=launch_height(mid);theta=-math.atan((launch_height(right)-launch_height(left))/.035)
   box((mid,0,z-.045),(.039,.86,.09),theta)
   for y in [-.46,.46]:box((mid,y,z+.045),(.039,.065,.18),theta)
  box((2.25,0,1.62),(.36,.74,.08),cfg_override=newton.ModelBuilder.ShapeConfig(density=0,mu=.08,restitution=.01))
  tray=builder.add_body(xform=wp.transform(wp.vec3(2.40,0,1.73),wp.quat_identity()))
  traycfg=newton.ModelBuilder.ShapeConfig(density=30,mu=.08,restitution=.01)
  for center,size in [((0,0,-.025),(.64,.66,.075)),((-.282,0,.12),(.075,.66,.29)),((.282,0,.12),(.075,.66,.29)),((0,-.292,.12),(.64,.075,.29)),((0,.292,.12),(.64,.075,.29))]:
   builder.add_shape_box(body=tray,xform=wp.transform(wp.vec3(*center),wp.quat_identity()),hx=size[0]/2,hy=size[1]/2,hz=size[2]/2,cfg=traycfg)
  for center,size in [((3.05,0,.25),(2.65,1.45,.12)),((1.77,0,.65),(.09,1.45,.8)),((4.33,0,.65),(.09,1.45,.8)),((3.05,.73,.65),(2.65,.09,.8)),((3.05,-.73,.65),(2.65,.09,.8))]:box(center,size)
  builder.add_ground_plane()
  sb=newton.ModelBuilder();SolverImplicitMPM.register_custom_attributes(sb)
  dx=a.particle_spacing;rng=np.random.default_rng(12);initial=[]
  for x in np.arange(a.sand_start,a.sand_end,dx):
   for y in np.arange(-a.sand_half_width,a.sand_half_width+dx*.1,dx):
    for h in np.arange(dx*.55,a.sand_thickness,dx):
     if True:
      q=np.array([x,y,1.755+h])+rng.uniform(-.001,.001,3);initial.append(q)
      sb.add_particle(pos=wp.vec3(q),vel=wp.vec3(0),mass=1500*dx**3,radius=dx/2,custom_attributes={'mpm:friction':a.sand_friction,'mpm:young_modulus':a.sand_young,'mpm:tensile_yield_ratio':0.,'mpm:yield_stress':0.})
  model=builder.finalize();sm=sb.finalize();options=SolverImplicitMPM.Config();options.voxel_size=a.voxel;options.grid_type=a.grid_type;options.grid_padding=8;options.max_active_cell_count=1<<17;options.strain_basis='P0';options.max_iterations=40;options.critical_fraction=0.
  mpm=SolverImplicitMPM(sm,config=options);mpm.setup_collider(model=model);rigid=SolverXPBD(model,iterations=15)
  state=model.state();other=model.state();ss=sm.state();ss.body_q=wp.empty_like(state.body_q);ss.body_qd=wp.empty_like(state.body_qd);ss.body_f=wp.empty_like(state.body_f)
  newton.eval_fk(model,model.joint_q,model.joint_qd,state)
  control=model.control();pipeline=newton.CollisionPipeline(model);contacts=pipeline.contacts();maxnodes=1<<18
  impulses=wp.zeros(maxnodes,dtype=wp.vec3);points=wp.zeros(maxnodes,dtype=wp.vec3);ids=wp.full(maxnodes,-1,dtype=int);forces=wp.zeros_like(state.body_f)
  def collect():
   f,x,c=mpm.collect_collider_impulses(ss);ids.fill_(-1);n=min(len(f),maxnodes);impulses[:n].assign(f[:n]);points[:n].assign(x[:n]);ids[:n].assign(c[:n])
  collect();dt=1/24/a.mpm_substeps
  def step(hold=False):
   nonlocal state,other
   for _ in range(4):
    state.clear_forces()
    wp.launch(compute_body_forces,dim=maxnodes,inputs=[dt,ids,impulses,points,mpm.collider_body_index,state.body_q,model.body_com,state.body_f]);forces.assign(state.body_f)
    pipeline.collide(state,contacts);rigid.step(state,other,control,contacts,dt/4);state,other=other,state
    if hold:
     held=state.body_q.numpy();held[0]=[-3.12,0,height(-3.12)+.185,0,0,0,1];state.body_q.assign(held);heldvel=state.body_qd.numpy();heldvel[0]=0;state.body_qd.assign(heldvel)
   wp.launch(subtract_body_force,dim=model.body_count,inputs=[dt,state.body_q,state.body_qd,forces,model.body_inv_inertia,model.body_inv_mass,ss.body_q,ss.body_qd])
   mpm.step(ss,ss,contacts=None,control=None,dt=dt);collect()
  for i in range(a.settle*a.mpm_substeps):
   step(hold=True)
   if i%10==0 and not np.isfinite(ss.particle_q.numpy()).all():
    raise RuntimeError(f'Nonfinite sand during settling substep {i}')
   if i%60==0:print('settle',i,'tray',state.body_q.numpy()[tray],flush=True)
  balls=[];quats=[];particles=[];vels=[];trayposes=[]
  from scipy.spatial.transform import Rotation
  retention=[]
  for i in range(a.frames):
   allposes=state.body_q.numpy();trayposes.append(allposes[tray].copy());q=allposes[0];sq=ss.particle_q.numpy()
   if not (np.isfinite(q).all() and np.isfinite(sq).all()):
    np.savez_compressed(out/'failure_diagnostic.npz',ball=np.asarray(balls),sand=np.asarray(particles),failed_sand=sq,failed_ball=q)
    (out/'failure.json').write_text(json.dumps({'frame':i,'nonfinite_particles':int((~np.isfinite(sq).all(axis=1)).sum()),'parameters':vars(a)},default=str,indent=2))
    raise RuntimeError(f'Nonfinite state at frame {i}; diagnostic saved')
   balls.append(q[:3].copy());quats.append(q[3:][[3,0,1,2]].copy());particles.append(sq.copy());vels.append(state.body_qd.numpy()[0].copy())
   local=Rotation.from_quat(allposes[tray,3:]).inv().apply(sq-allposes[tray,:3])
   below=(local[:,2]<-.015);side=(np.abs(local[:,0])>.322)|(np.abs(local[:,1])>.332)
   retention.append({'frame':i,'below_floor':int(below.sum()),'outside_outer_sides':int(side.sum())})
   (out/'retention.json').write_text(json.dumps(retention,indent=2))
   if i%6==0:print('retention',retention[-1],flush=True)
   if i%6==0:print('frame',i,'ball',balls[-1],'elapsed',time.time()-started,flush=True)
   for _ in range(a.mpm_substeps):step()
  b=np.asarray(balls);q=np.asarray(particles);events={}
  for name,x in [('smoke',-2.1),('cloth',-.8),('sand',a.sand_start),('ramp_exit',1.7),('tank',1.85)]:
   indices=np.where(b[:,0]>=x)[0];events[name]=int(indices[0]) if len(indices) else None
  assert np.isfinite(q).all() and np.isfinite(b).all()
  np.savez_compressed(out/'rigid.npz',ball=b,quaternions=np.asarray(quats),sand=q,t=np.arange(a.frames)/24,ball_velocities=np.asarray(vels),tray_poses=np.asarray(trayposes))
  meta={'backend':'Newton 1.7.0 / XPBD + implicit MPM','frames':a.frames,'fps':24,'ball_radius':.18,'ball_density':500,'grain_radius':dx/2,'simulation_particles':len(q[0]),'mpm_voxel_size':a.voxel,'material':{'friction':a.sand_friction,'young_modulus_pa':a.sand_young,'tensile_yield_ratio':0.,'yield_stress_pa':0.},'particle_spacing':dx,'layout':'curved launch lip; dynamic five-box shallow tray on platform, XPBD and MPM impulse coupling; upstream unchanged','sand_geometry':{'start_x':a.sand_start,'end_x':a.sand_end,'maximum_thickness_m':a.sand_thickness,'half_width_m':a.sand_half_width},'events':events,'elapsed_s':time.time()-started,'device':a.device,'versions':{'newton':'1.7.0','warp':wp.__version__},'coupling':'Official two-way MPM collider impulse exchange to XPBD. Other solvers receive trajectory one-way. MPM material samples are not individual grains.'}
  (out/'rigid.json').write_text(json.dumps(meta,indent=2));print(meta,flush=True)
  if a.frames>69:assert events['tank'] is not None,(b[-1],events)
if __name__=='__main__':main()
