"""Bake separate cloth and Mantaflow responses to the same solved ball trajectory."""
import bpy,sys,json,time,hashlib
from pathlib import Path
sys.path.append('/usr/lib/python3/dist-packages')
import numpy as np
args=sys.argv[sys.argv.index('--')+1:];root=Path(args[0]).resolve();kind=args[1];out=root/kind;out.mkdir(exist_ok=True)
a=np.load(root/'rigid.npz');s=bpy.context.scene
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
s.frame_start=1;s.frame_end=124;s.render.fps=24
bpy.ops.mesh.primitive_uv_sphere_add(segments=24,ring_count=16,radius=.18);ball=bpy.context.object;ball.name='trajectory_ball'
for i,p in enumerate(a['ball']):ball.location=p;ball.keyframe_insert('location',frame=i+1)
for fc in ball.animation_data.action.fcurves:
 for k in fc.keyframe_points:k.interpolation='LINEAR'
def box(name,loc,size):
 bpy.ops.mesh.primitive_cube_add(size=1,location=loc);o=bpy.context.object;o.name=name;o.dimensions=size;bpy.ops.object.transform_apply(location=False,rotation=False,scale=True);return o
if kind=='cloth':
 ball.modifiers.new('ball collision','COLLISION');ball.collision.thickness_outer=.006
 n=25;verts=[(-.8,-.34+j*.68/(n-1),2.6-i*1.0/(n-1)) for i in range(n) for j in range(n)];faces=[]
 for i in range(n-1):
  for j in range(n-1):
   k=i*n+j;faces.append((k,k+1,k+n+1,k+n))
 m=bpy.data.meshes.new('cloth');m.from_pydata(verts,[],faces);m.update();o=bpy.data.objects.new('curtain',m);s.collection.objects.link(o)
 pins=o.vertex_groups.new(name='top_pinned');pins.add(list(range(n)),1,'REPLACE')
 mod=o.modifiers.new('physical cloth','CLOTH');c=mod.settings;c.quality=8;c.mass=.08;c.tension_stiffness=25;c.compression_stiffness=25;c.shear_stiffness=15;c.bending_stiffness=.06;c.vertex_group_mass=pins.name;c.pin_stiffness=1
 mod.collision_settings.use_collision=True;mod.collision_settings.distance_min=.006;mod.collision_settings.collision_quality=5;mod.collision_settings.use_self_collision=True;mod.collision_settings.self_distance_min=.006
 mod.point_cache.frame_start=1;mod.point_cache.frame_end=124
 q=[]
 for f in range(1,125):
  s.frame_set(f);dg=bpy.context.evaluated_depsgraph_get();ev=o.evaluated_get(dg);mesh=ev.to_mesh();q.append([list(v.co) for v in mesh.vertices]);ev.to_mesh_clear()
  if f%24==0:print('cloth',f,flush=True)
 tri=[]
 for f in faces:tri.extend([(f[0],f[1],f[2]),(f[0],f[2],f[3])])
 np.savez_compressed(out/'states.npz',positions=np.asarray(q),triangles=np.asarray(tri),t=np.arange(124)/24)
 (out/'simulation.json').write_text(json.dumps({'backend':'Blender cloth mass-spring solver','frames':124,'fps':24,'vertices':n*n,'coupling':'one-way solved ball trajectory drives cloth collision; reaction not returned to rigid solver'},indent=2))
else:
 f=ball.modifiers.new('moving fluid obstacle','FLUID');f.fluid_type='EFFECTOR';f.effector_settings.surface_distance=.001
 if kind=='liquid':
  domain=box('liquid domain',(3.05,0,1.04),(2.7,1.5,1.48));source=box('initial water',(3.05,0,.565),(2.52,1.34,.49))
  fm=source.modifiers.new('initial water','FLUID');fm.fluid_type='FLOW';fs=fm.flow_settings;fs.flow_type='LIQUID';fs.flow_behavior='GEOMETRY';source.hide_render=True
 else:
  domain=box('smoke domain',(-2.1,0,2.55),(1.1,1.25,1.8));bpy.ops.mesh.primitive_uv_sphere_add(segments=12,ring_count=8,radius=.105,location=(-2.1,0,1.88));source=bpy.context.object
  fm=source.modifiers.new('smoke source','FLUID');fm.fluid_type='FLOW';fs=fm.flow_settings;fs.flow_type='SMOKE';fs.flow_behavior='INFLOW';fs.density=.8;fs.temperature=1.5;source.hide_render=True
 mod=domain.modifiers.new('domain','FLUID');mod.fluid_type='DOMAIN';ds=mod.domain_settings;ds.domain_type='LIQUID' if kind=='liquid' else 'GAS';ds.resolution_max=56 if kind=='liquid' else 32;ds.cache_type='MODULAR';ds.cache_frame_start=1;ds.cache_frame_end=124;ds.cache_directory=str(out/'cache');ds.timesteps_max=8
 if kind=='liquid':ds.use_mesh=True;ds.mesh_scale=2
 bpy.context.view_layer.objects.active=domain;bpy.ops.object.select_all(action='DESELECT');domain.select_set(True)
 bpy.ops.wm.save_as_mainfile(filepath=str(out/'scene.blend'));print('baking',kind,flush=True);bpy.ops.fluid.bake_data()
 if kind=='liquid':
  bpy.ops.fluid.bake_mesh();surfaces=out/'surfaces';surfaces.mkdir(exist_ok=True)
  for f in range(1,125):
   s.frame_set(f);dg=bpy.context.evaluated_depsgraph_get();ev=domain.evaluated_get(dg);mesh=ev.to_mesh();mesh.calc_loop_triangles();v=np.array([list(ev.matrix_world@x.co) for x in mesh.vertices]);faces=np.array([list(x.vertices) for x in mesh.loop_triangles]);np.savez_compressed(surfaces/f'{f-1:04d}.npz',vertices=v,faces=faces);ev.to_mesh_clear()
   if f%24==0:print('mesh',f,len(v),flush=True)
 bpy.ops.wm.save_as_mainfile(filepath=str(out/'scene.blend'))
 (out/'simulation.json').write_text(json.dumps({'backend':'Blender Mantaflow '+kind,'frames':124,'fps':24,'resolution':ds.resolution_max,'source_trajectory_sha256':hashlib.sha256((root/'rigid.npz').read_bytes()).hexdigest(),'coupling':'one-way kinematic obstacle from solved sphere trajectory; no fluid forces returned to ball'},indent=2))
print('DONE',kind,flush=True)
