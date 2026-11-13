import bpy,json,math
from pathlib import Path
from mathutils import Vector,Quaternion
p=Path(__file__).parent;d=json.loads((p/'render_data.json').read_text());bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
s=bpy.context.scene;s.render.engine='BLENDER_EEVEE';s.eevee.taa_render_samples=16;s.render.resolution_x=768;s.render.resolution_y=512;s.render.resolution_percentage=100;s.render.fps=24;s.world.color=(.2,.2,.2)
def material(n,c):
 m=bpy.data.materials.new(n);m.diffuse_color=(*c,1);return m
iv=material('ivory',(.8,.74,.6));eb=material('ebony',(.055,.04,.025));skin=material('MANO',(.66,.46,.33))
def box(loc,sc,m):
 bpy.ops.mesh.primitive_cube_add(size=1,location=loc);o=bpy.context.object;o.scale=sc;o.data.materials.append(m);return o
box((0,0,-.01),(.35,.35,.02),eb)
for y in range(8):
 for x in range(8):box(((x-3.5)*.04,(y-3.5)*.04,.0002),(.04,.04,.0004),iv if (x+y)%2 else eb)
objects=[]
for side in range(2):
 for rank in range(2):
  for c in range(8):
   kind='pawn' if rank else ['rook','knight','bishop','queen','king','bishop','knight','rook'][c];h=.045 if rank else [.055,.060,.065,.075,.080,.065,.060,.055][c]
   profile=[(0,.012),(.003,.012),(.006,.011),(.009,.008),(.012,.007),(h*.55,.004),(h*.68,.006),(h*.73,.007),(h*.78,.005),(h*.84,.007),(h*.94,.008),(h,.001)]
   verts=[];faces=[];N=64
   for z,r in profile:
    for i in range(N):a=2*math.pi*i/N;verts.append((r*math.cos(a),r*math.sin(a),z))
   for j in range(len(profile)-1):
    for i in range(N):a=j*N+i;b=j*N+(i+1)%N;faces.append((a,b,b+N,a+N))
   me=bpy.data.meshes.new(kind);me.from_pydata(verts,[],faces);me.update();o=bpy.data.objects.new(f'{side}_{kind}_{c}',me);s.collection.objects.link(o);o.data.materials.append(iv if side==0 else eb);objects.append(o)
   for poly in me.polygons:poly.use_smooth=True
   # Distinct crowns, crosses, bishop split and horse-head silhouette.
   if kind=='king':
    for loc,sc in [((0,0,h+.004),(.003,.003,.014)),((0,0,h+.007),(.012,.003,.003))]:a=box(loc,sc,o.data.materials[0]);a.parent=o
   if kind in ['rook','queen']:
    for i in range(6):a=i*math.pi/3;o2=box((.006*math.cos(a),.006*math.sin(a),h),(.004,.004,.006),o.data.materials[0]);o2.parent=o;o2.data.materials.append(o.data.materials[0])
   if kind=='knight':
    a=box((0,-.003,h),(.010,.016,.017),o.data.materials[0]);a.rotation_euler.x=-.35;a.parent=o
bpy.ops.object.light_add(type='AREA',location=(.1,-.3,.7));bpy.context.object.data.energy=18;bpy.context.object.data.size=.5
bpy.ops.object.camera_add(location=(.42,-.55,.48));cam=bpy.context.object;cam.rotation_euler=(Vector((0,0,.035))-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.type='ORTHO';cam.data.ortho_scale=.49;s.camera=cam
(p/'frames').mkdir(exist_ok=True)
for i,q in enumerate(d['qpos']):
 for j,o in enumerate(objects):o.location=q[j*7:j*7+3];o.rotation_mode='QUATERNION';o.rotation_quaternion=q[j*7+3:j*7+7]
 s.render.filepath=str(p/'frames'/f'{i:04d}.png');bpy.ops.render.render(write_still=True)
