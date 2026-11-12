"""Render actual reconstructed/cloth surfaces in a fully visible fixed view."""
import json
import sys
import time
from pathlib import Path

import bpy
sys.path.append('/usr/lib/python3/dist-packages')
import numpy as np
from mathutils import Vector

args = sys.argv[sys.argv.index('--')+1:]
root = Path(args[0])
meta = json.loads((root/'simulation.json').read_text())
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE'
scene.eevee.taa_render_samples = 32
scene.eevee.use_gtao = True
scene.eevee.gtao_distance = .3
scene.render.resolution_x = 768
scene.render.resolution_y = 512
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'
scene.render.fps = 24
scene.world.color = (.32, .32, .32)
scene.view_settings.view_transform = 'Standard'


def material(name, value, roughness=.5):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (*value, 1)
    bsdf.inputs['Roughness'].default_value = roughness
    return m


clay = material('neutral white physical surface', (.72, .74, .76), .28)
walls = material('grey channel or sphere', (.22, .24, .27), .6)
ground = material('neutral floor', (.14, .15, .17), .7)
bpy.ops.mesh.primitive_plane_add(size=200)
bpy.context.object.data.materials.append(ground)
cloth = meta['case'] == 'cloth_bspline'
bpy.context.object.location.z = -1.2 if cloth else -.085
for location, energy, size in [((1, -4, 7), 1500, 5), ((-4, 1, 5), 700, 4)]:
    bpy.ops.object.light_add(type='AREA', location=location)
    bpy.context.object.data.energy = energy
    bpy.context.object.data.size = size
bpy.ops.object.camera_add(location=(4, -7, 3.8))
cam = bpy.context.object
focus = Vector((0, 0, .15 if cloth else .6))
cam.rotation_euler = (focus-cam.location).to_track_quat('-Z', 'Y').to_euler()
cam.data.type = 'ORTHO'
cam.data.ortho_scale = 5.8 if cloth else 4.4
scene.camera = cam
if cloth:
    # Display the same geometry used for collision. A smooth analytic sphere
    # extends beyond an inscribed icosphere and falsely appears to pierce cloth.
    collider_vertices, collider_faces = [], []
    for line in (root/'models/sphere.obj').read_text().splitlines():
        if line.startswith('v '):
            collider_vertices.append([float(x) for x in line.split()[1:4]])
        elif line.startswith('f '):
            collider_faces.append([int(x.split('/')[0])-1 for x in line.split()[1:4]])
    collider_mesh = bpy.data.meshes.new('actual collision sphere')
    collider_mesh.from_pydata(collider_vertices, [], collider_faces)
    sphere = bpy.data.objects.new('actual collision sphere', collider_mesh)
    scene.collection.objects.link(sphere)
    collider_mesh.materials.append(walls)
    for p in collider_mesh.polygons:
        p.use_smooth = True
    for point in meta['pinned_positions']:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=8, radius=.022, location=point)
        bpy.context.object.data.materials.append(walls)
else:
    for loc, scale in [((0, 0, -.04), (1.3, .5, .04)),
                       ((-1.34, 0, .75), (.04, .5, .8)),
                       ((1.34, 0, .75), (.04, .5, .8)),
                       ((0, .54, .75), (1.3, .04, .8))]:
        bpy.ops.mesh.primitive_cube_add(size=2, location=loc)
        bpy.context.object.scale = scale
        bpy.context.object.data.materials.append(walls)
mesh = bpy.data.meshes.new('actual simulated surface')
obj = bpy.data.objects.new('physical surface', mesh)
scene.collection.objects.link(obj)
mesh.materials.append(clay)
frames = root/'frames'
frames.mkdir(exist_ok=True)
start = int(args[args.index('--start')+1]) if '--start' in args else 0
end = int(args[args.index('--end')+1]) if '--end' in args else meta['frames']
indices = [0, 12, 24, 48] if '--preview' in args else range(start, end)
for index in indices:
    if '--resume' in args and (frames/f'{index:04d}.png').is_file():
        continue
    surface_path = root/'surfaces'/f'{index:04d}.npz'
    deadline = time.time()+1800
    while not surface_path.is_file() and '--wait' in args:
        if time.time() > deadline:
            raise TimeoutError(surface_path)
        time.sleep(.5)
    with np.load(surface_path) as data:
        mesh.clear_geometry()
        mesh.from_pydata(data['vertices'].tolist(), [], data['faces'].tolist())
    mesh.update()
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    scene.render.filepath = str(frames/f'{index:04d}.png')
    bpy.ops.render.render(write_still=True)
    if index % 12 == 0:
        print('rendered', index, flush=True)
(root/'render.json').write_text(json.dumps({'engine': 'Blender EEVEE', 'size': [768, 512],
    'frames': meta['frames'], 'fps': 24, 'camera_position': list(cam.location),
    'camera_focus': list(focus), 'ortho_scale': cam.data.ortho_scale,
    'geometry': 'Actual solver surface samples; smooth normals only, no subdivision deformation.'}, indent=2))
