"""DFSPH water tank with Newton trajectory as a moving sphere boundary."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import pysplishsplash as sph
import pysplashsurf,trimesh
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--radius',type=float,default=.018);p.add_argument('--frames',type=int,default=124);a=p.parse_args();root=a.output.resolve();out=root/'liquid';out.mkdir(parents=True,exist_ok=True)
data=np.load(root/'rigid.npz');times=data['t'];ball=data['ball'];dt=1/24/24;frames=min(a.frames,len(times));start=time.time()
unit=Path(__file__).resolve().parents[1]/'vendor/UnitBox.obj';sphere=trimesh.creation.icosphere(subdivisions=3,radius=.18);sphere.export(out/'sphere.obj')
scene={'Configuration':{'particleRadius':a.radius,'simulationMethod':4,'gravitation':[0,-9.81,0],'cflMethod':0,'timeStepSize':dt,'boundaryHandlingMethod':2,'enableZSort':False,'stopAt':(frames-1)/24-dt*.5},'Materials':[{'id':'Fluid','density0':1000,'viscosityMethod':1,'Standard viscosity':{'viscosity':.005}}],'RigidBodies':[{'geometryFile':str(unit),'translation':[3.05,.70,0],'scale':[2.56,.78,1.36],'isDynamic':False,'isWall':True,'mapInvert':True,'mapResolution':[40,40,40]},{'geometryFile':str(out/'sphere.obj'),'translation':ball[0][[0,2,1]].tolist(),'scale':[1.,1.,1.],'isDynamic':False,'isAnimated':True,'isWall':False,'mapInvert':False,'mapResolution':[30,30,30]}],'FluidBlocks':[{'denseMode':0,'start':[1.84,.35,-.63],'end':[4.26,.79,.63]}]}
(out/'scene.json').write_text(json.dumps(scene,indent=2));base=sph.Exec.SimulatorBase();base.init(sceneFile=str(out/'scene.json'),useGui=False,initialPause=False,outputDir=str(out/'sph_cache'));base.initSimulation();print('simulation initialized',flush=True);sim=sph.Simulation.getCurrent();fluid=sim.getFluidModel(0);tm=sph.TimeManager.getCurrent();tm.setTimeStepSize(dt);count=fluid.numActiveParticles();rbo=None
surfaces=out/'surfaces';surfaces.mkdir(exist_ok=True);positions=[];records=[];boundary_records=[]
def pose(t):
 pos=np.array([np.interp(t,times,ball[:,k]) for k in range(3)]);j=min(max(np.searchsorted(times,t,side='right')-1,0),len(times)-2);velocity=(ball[j+1]-ball[j])/(times[j+1]-times[j])
 if t>=times[-1]:velocity*=0
 return pos,velocity
def update_boundary(t):
 global rbo
 if rbo is None:rbo=sim.getBoundaryModel(1).getRigidBodyObject();rbo.setIsAnimated(True)
 pos,vel=pose(t);rbo.setPosition(pos[[0,2,1]]);rbo.setVelocity(vel[[0,2,1]]);rbo.setAngularVelocity(np.zeros(3));rbo.updateMeshTransformation();return pos
def capture(i):
 q=np.asarray([fluid.getPosition(j) for j in range(count)],dtype=np.float32)[:,[0,2,1]];assert np.isfinite(q).all();positions.append(q)
 md,_=pysplashsurf.reconstruction_pipeline(np.ascontiguousarray(q,dtype=np.float64),particle_radius=a.radius,rest_density=1000.,smoothing_length=2.,cube_size=.6,iso_surface_threshold=.6,mesh_smoothing_weights=True,mesh_smoothing_iters=5,mesh_cleanup=True,compute_normals=False,subdomain_grid=True,subdomain_num_cubes_per_dim=64)
 vertices,faces=np.asarray(md.mesh.vertices),np.asarray(md.mesh.triangles);assert len(vertices) and np.isfinite(vertices).all();np.savez_compressed(surfaces/f'{i:04d}.npz',vertices=vertices.astype('f'),faces=faces.astype('i'))
 actual=np.asarray(rbo.getPosition())[[0,2,1]] if rbo is not None else ball[0];boundary_records.append(actual.tolist());records.append({'frame':i,'time':i/24,'boundary_error_m':float(np.linalg.norm(actual-ball[i])),'surface_vertices':len(vertices)})
 if i%6==0:print('DFSPH frame',i,'particles',count,'elapsed',time.time()-start,flush=True)
capture(0);step=0
def callback():
 global step
 step+=1;t=step*dt;tm.setTime(t);tm.setTimeStepSize(dt);update_boundary(t)
 if step%24==0:capture(step//24)
base.setTimeStepCB(callback);base.runSimulation();assert len(positions)==frames
q=np.asarray(positions);assert np.isfinite(q).all();np.savez_compressed(out/'states.npz',positions=q,t=np.arange(frames)/24,boundary_positions=np.asarray(boundary_records))
lo=q.min(axis=(0,1));hi=q.max(axis=(0,1));checks={'finite':True,'bounds_min':lo.tolist(),'bounds_max':hi.tolist(),'max_boundary_trajectory_error_m':max(x['boundary_error_m'] for x in records),'particle_count':count,'inside_horizontal_tank_bounds_with_5cm_tolerance':bool(np.all(lo[:2]>=np.array([1.77,-.73])-.05) and np.all(hi[:2]<=np.array([4.33,.73])+.05))}
meta={'backend':'SPlisHSPlasH 2.19.1 / DFSPH','surface':'splashsurf 0.14.1.0','radius':a.radius,'frames':frames,'fps':24,'substeps_per_frame':24,'elapsed_s':time.time()-start,'source_trajectory_sha256':hashlib.sha256((root/'rigid.npz').read_bytes()).hexdigest(),'clock':{'time_step_s':dt,'output_times':'exact frame index / 24; internal float32 clock reset per step'},'coupling':'Newton ball trajectory drives animated DFSPH sphere boundary. Fluid reaction is not returned to ball; MPM sand is not a fluid boundary.','checks':checks};(out/'simulation.json').write_text(json.dumps(meta,indent=2));(out/'surface_manifest.json').write_text(json.dumps(records,indent=2));print(meta,flush=True);base.cleanup()
