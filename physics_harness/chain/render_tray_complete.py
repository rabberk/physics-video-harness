"""Local clay preview of solved ball/MPM samples; water level is a static marker."""
import bpy,sys,math,json
import numpy as np
from pathlib import Path
from mathutils import Vector
args=sys.argv[sys.argv.index('--')+1:];with_water='--water' in args;root=Path(args[0]).resolve();d=np.load(root/'rigid.npz');q=d['sand'];scene=bpy.context.scene
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
def mat(name,c):
 m=bpy.data.materials.new(name);m.diffuse_color=(*c,1);return m
clay=mat('ball clay',(.8,.81,.82));sandmat=mat('MPM sample display',(.48,.29,.10));trackmat=mat('track',(.3,.33,.36));watermat=mat('static water level marker',(.035,.24,.42))
def box(loc,size,material,angle=0):
 bpy.ops.mesh.primitive_cube_add(size=1,location=loc);o=bpy.context.object;o.dimensions=size;o.rotation_euler[1]=angle;o.data.materials.append(material);return o
def height(x):return 1.64-.2*x
box((-1.325,0,height(-1.325)-.06),(4.45,.86,.12),trackmat,math.atan(.2))
box((-1.325,.46,height(-1.325)+.055),(4.45,.065,.22),trackmat,math.atan(.2))
def launch_height(x):return height(x)+.55*max(0.,x-.85)**2
for left in np.linspace(.85,1.62,23):
 right=left+.035;mid=(left+right)/2;z=launch_height(mid);theta=-math.atan((launch_height(right)-launch_height(left))/.035)
 box((mid,0,z-.045),(.039,.86,.09),trackmat,theta)
 box((mid,.46,z+.045),(.039,.065,.18),trackmat,theta)
box((2.25,0,1.62),(.36,.74,.08),trackmat)
trayparts=[]
for center,size in [((0,0,-.025),(.64,.66,.075)),((-.282,0,.12),(.075,.66,.29)),((.282,0,.12),(.075,.66,.29)),((0,-.292,.12),(.64,.075,.29)),((0,.292,.12),(.64,.075,.29))]:
 o=box((0,0,0),size,clay);trayparts.append((o,Vector(center)))
for loc,size in [((3.05,0,.25),(2.65,1.45,.12)),((4.33,0,.65),(.09,1.45,.8)),((3.05,.73,.65),(2.65,.09,.8)),((1.77,0,.65),(.09,1.45,.8))]:box(loc,size,trackmat)
# Far-half marker leaves falling samples visible; no dynamic liquid is claimed.
if not with_water:box((3.05,.40,.785),(2.42,.48,.01),watermat)
water=None
if with_water:
 wm=bpy.data.meshes.new('DFSPH liquid surface');water=bpy.data.objects.new('DFSPH water',wm);scene.collection.objects.link(water);wm.materials.append(watermat)
cloth=np.load(root/'cloth/states.npz')
cm=bpy.data.meshes.new('BS-Cloth surface');sheet=bpy.data.objects.new('BS-Cloth curtain',cm);scene.collection.objects.link(sheet);cm.materials.append(clay)
for y in [-.38,.38]:box((-.8,y,2.1),(.035,.035,1.1),trackmat)
box((-.8,0,2.62),(.05,.82,.05),trackmat)
bpy.ops.mesh.primitive_cylinder_add(vertices=24,radius=.12,depth=.3,location=(-2.1,0,1.77));bpy.context.object.data.materials.append(trackmat)
bpy.ops.object.volume_import(filepath=str(root/'smoke/cache/data/fluid_data_0001.vdb'));smoke=bpy.context.object;smoke.location=(-2.65,-.625,1.65)
sm=bpy.data.materials.new('dark smoke');sm.use_nodes=True;nodes=sm.node_tree.nodes;nodes.clear();volume=nodes.new('ShaderNodeVolumePrincipled');volume.inputs['Density'].default_value=10.;volume.inputs['Color'].default_value=(.12,.14,.16,1);output=nodes.new('ShaderNodeOutputMaterial');sm.node_tree.links.new(volume.outputs['Volume'],output.inputs['Volume']);smoke.data.materials.append(sm)
box((2,0,-.09),(200,200,.1),trackmat)
bpy.ops.mesh.primitive_uv_sphere_add(segments=40,ring_count=24,radius=.18);ball=bpy.context.object;ball.data.materials.append(clay)
for p in ball.data.polygons:p.use_smooth=True
# One octahedron per material sample, radius set to half the material spacing.
r=json.loads((root/'rigid.json').read_text())['particle_spacing']/2;base=np.array([[r,0,0],[-r,0,0],[0,r,0],[0,-r,0],[0,0,r],[0,0,-r]],dtype='f');f=np.array([[0,2,4],[2,1,4],[1,3,4],[3,0,4],[2,0,5],[1,2,5],[3,1,5],[0,3,5]])
faces=(f[None,:,:]+6*np.arange(q.shape[1])[:,None,None]).reshape(-1,3);mesh=bpy.data.meshes.new('material sample geometry');mesh.from_pydata((q[0,:,None,:]+base).reshape(-1,3).tolist(),[],faces.tolist());obj=bpy.data.objects.new('actual MPM samples',mesh);scene.collection.objects.link(obj);mesh.materials.append(sandmat)
for loc,power in [((1,-4,7),1500),((4,3,6),1100)]:
 bpy.ops.object.light_add(type='AREA',location=loc);bpy.context.object.data.energy=power;bpy.context.object.data.size=5
bpy.ops.object.camera_add(location=(5.8,-11.8,7.1));cam=bpy.context.object;cam.rotation_euler=(Vector((.4,0,1.25))-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.type='ORTHO';cam.data.ortho_scale=9.1;scene.camera=cam
scene.render.engine='CYCLES';scene.cycles.samples=12;scene.cycles.device='GPU';prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='CUDA';prefs.get_devices()
for dev in prefs.devices:dev.use=dev.type=='CUDA'
scene.render.resolution_x=1280;scene.render.resolution_y=720;scene.render.resolution_percentage=100;scene.world.color=(.3,.3,.3);scene.render.fps=24
out=root/('complete_frames' if with_water else 'complete_frames_no_water');out.mkdir(exist_ok=True)
start=int(args[args.index('--start')+1]) if '--start' in args else 0
end=int(args[args.index('--end')+1]) if '--end' in args else 124
for i in ([0] if '--still' in args else range(start,end)):
 smoke.data.is_sequence=False;smoke.data.filepath=str(root/'smoke/cache/data'/f'fluid_data_{i+1:04d}.vdb')
 sheet.data.clear_geometry();sheet.data.from_pydata(cloth['positions'][i].tolist(),[],cloth['triangles'].tolist());sheet.data.update()
 for face in sheet.data.polygons:face.use_smooth=True
 if with_water:
  surf=np.load(root/'liquid/surfaces'/f'{i:04d}.npz');water.data.clear_geometry();water.data.from_pydata(surf['vertices'].tolist(),[],surf['faces'].tolist());water.data.update()
  for face in water.data.polygons:face.use_smooth=True
 pose=d['tray_poses'][i]
 from mathutils import Quaternion
 rot=Quaternion((pose[6],pose[3],pose[4],pose[5]))
 for ob,offset in trayparts:ob.location=Vector(pose[:3])+rot@offset;ob.rotation_mode='QUATERNION';ob.rotation_quaternion=rot
 ball.location=d['ball'][i];mesh.vertices.foreach_set('co',(q[i,:,None,:]+base).reshape(-1).astype('f'));mesh.update();scene.render.filepath=str(out/f'{i:04d}.png');bpy.ops.render.render(write_still=True);print('FRAME',i,flush=True)
(root/(f'complete_render_{start}_{end}.json')).write_text(json.dumps({'source_frames':[0,123],'fps':24,'water':('DFSPH driven by current ball trajectory; no sand-water or fluid-to-ball feedback' if with_water else 'static far-half water level marker, no fluid coupling'),'sand':'octahedra at actual MPM sample positions; samples are not individual grains','size':[1280,720]},indent=2))
