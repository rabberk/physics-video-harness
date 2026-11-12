import argparse,json,hashlib
from pathlib import Path
import numpy as np,yaml
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.output.resolve();out=root/'cloth';data=np.load(root/'rigid.npz');n=len(data['t']);positions=[];surfaces=out/'surfaces';surfaces.mkdir(exist_ok=True)
for i in range(n):
 verts=[];faces=[]
 for line in (out/'preview_meshes'/f'frame_{i}.obj').read_text().splitlines():
  if line.startswith('v '):verts.append([float(x) for x in line.split()[1:4]])
  elif line.startswith('f '):faces.append([int(x.split('/')[0])-1 for x in line.split()[1:4]])
 v=np.asarray(verts);f=np.asarray(faces);assert np.isfinite(v).all();positions.append(v);np.savez_compressed(surfaces/f'{i:04d}.npz',vertices=v,faces=f)
q=np.asarray(positions);pins=np.where(q[0,:,2]>q[0,:,2].max()-1e-6)[0];pin_displacement=float(np.linalg.norm(q[:,pins]-q[0,pins],axis=2).max());assert pin_displacement<1e-4
clearance=np.linalg.norm(q-data['ball'][:,None,:],axis=2)-.18
if clearance.min() < -.015:raise RuntimeError(f'Cloth analytic-sphere penetration too large: {clearance.min()} m')
np.savez_compressed(out/'states.npz',positions=q,triangles=f,t=data['t']);effective=yaml.safe_load((out/'steps/start.yaml').read_text());requested=yaml.safe_load((out/'config.yaml').read_text())
meta={'backend':'BS-Cloth / B-spline FEM + IPC','frames':n,'fps':24,'source_trajectory_sha256':hashlib.sha256((root/'rigid.npz').read_bytes()).hexdigest(),'coupling':'Newton trajectory drives native animated IPC collision mesh; reaction not returned','checks':{'finite':True,'pinned_max_displacement_m':pin_displacement,'max_displacement_m':float(np.linalg.norm(q-q[:1],axis=2).max()),'minimum_sample_distance_to_analytic_ball_surface_m':float(clearance.min()),'note':'Collision mesh is faceted, analytic sphere distance may be slightly negative. Sampled checks do not establish triangle nonintersection.'},'substeps_per_frame':round(1/24/requested['timestep']),'configuration_audit':{k:{'requested':requested.get(k,'automatic'),'serialized':effective.get(k,'automatic')} for k in ['stretching_youngs_modulus','bending_youngs_modulus','contact_stiffness','timestep']}}
(out/'simulation.json').write_text(json.dumps(meta,indent=2));print(meta,flush=True)
