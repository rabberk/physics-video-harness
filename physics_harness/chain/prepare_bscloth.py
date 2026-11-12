"""Adapt Newton's solved ball to BS-Cloth's native animated collision mesh."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,yaml,trimesh
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.output.resolve();out=root/'cloth';models=out/'models';anim=models/'ball';anim.mkdir(parents=True,exist_ok=True)
data=np.load(root/'rigid.npz');times=data['t'];ball=data['ball'];n=12;substeps=5;dt=1/24/substeps;steps=(len(times)-1)*substeps
sphere=trimesh.creation.icosphere(subdivisions=2,radius=.18)
for i in range(steps+1):
 pos=np.array([np.interp((i+1)*dt,times,ball[:,k]) for k in range(3)])
 mesh=sphere.copy();mesh.apply_translation(pos);mesh.export(anim/f'pose_{i}.obj')
track=[];angle=np.arctan(.2)
def box(center,size,rotation=0):
 m=trimesh.creation.box(extents=size);m.apply_transform(trimesh.transformations.rotation_matrix(rotation,[0,1,0]));m.apply_translation(center);track.append(m)
def height(x):return 1.64-.2*x
box((-.9,0,height(-.9)-.06),(5.3,.86,.12),angle)
# One closed deck mesh avoids static-static intersections at rail/deck overlap.
# Rails remain Newton colliders and visible geometry; cloth deck collision is retained.
trimesh.util.concatenate(track).export(models/'track.obj')
config=yaml.safe_load((Path(__file__).resolve().parents[2]/'outputs/sim_quality_20261111/cloth_bspline_v2/config.yaml').read_text())
config.update(max_steps=steps,timestep=dt,stretching_youngs_modulus=2000000.,bending_youngs_modulus=100.,ground_height=-.05,contact_threshold=.0002,contact_stiffness=0.,max_newton_iterations=40,residue_tolerance=.01)
config.pop('contact_stiffness',None)
config['disable_interpatch_contact']=True
config['bs_surfaces']=[{'resolution':{'x':n,'y':n},'linear_resolution':{'x':37,'y':37},'rest_size':{'x':.68,'y':.8},'fixed_nodes':list(range(n*(n-1),n*n)),'transformations':[{'type':'rotation','content':[90.,0.,90.]},{'type':'translation','content':[-.8,-.34,1.815]}]}]
config['fixed_trig_meshes']=[{'name':'track'}];config['animated_meshes']={'directory':'ball','pattern':'pose_@','length':steps+1,'stiffness':1.e12,'diff':.00001}
(out/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False));(out/'adapter.json').write_text(json.dumps({'solver':'BS-Cloth','source_trajectory_sha256':hashlib.sha256((root/'rigid.npz').read_bytes()).hexdigest(),'time_step':dt,'substeps_per_frame':substeps,'collision_target_clock':'pose k targets end of step k+1','collision_sphere_vertices':len(sphere.vertices),'coupling':'Newton trajectory -> animated IPC collision mesh; cloth forces not returned to Newton'},indent=2))
print('BS-Cloth prepared',out,flush=True)
