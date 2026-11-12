"""Pack BS-Cloth's evaluated surface samples without changing dynamics."""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml


def read_obj(path):
    vertices, faces = [], []
    for line in path.read_text().splitlines():
        if line.startswith('v '):
            vertices.append([float(x) for x in line.split()[1:4]])
        elif line.startswith('f '):
            faces.append([int(x.split('/')[0])-1 for x in line.split()[1:4]])
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    config = yaml.safe_load((out/'config.yaml').read_text())
    effective = yaml.safe_load((out/'steps/start.yaml').read_text())
    frames = config['max_steps']//5+1
    surfaces = out/'surfaces'
    surfaces.mkdir(exist_ok=True)
    positions = []
    for index in range(frames):
        source_path = out/'preview_meshes'/f'frame_{index}.obj'
        next_path = out/'preview_meshes'/f'frame_{index+1}.obj'
        deadline = time.time()+1800
        while args.wait and (not source_path.is_file() or (index+1 < frames and not next_path.is_file())):
            if time.time() > deadline:
                raise TimeoutError(source_path)
            time.sleep(.5)
        vertices, faces = read_obj(source_path)
        assert len(vertices) and len(faces) and np.isfinite(vertices).all()
        pending = surfaces/f'{index:04d}.pending.npz'
        np.savez_compressed(pending, vertices=vertices, faces=faces)
        pending.replace(surfaces/f'{index:04d}.npz')
        positions.append(vertices)
    p = np.asarray(positions)
    first = p[0]
    pins = [int(np.argmin(np.linalg.norm(first-point, axis=1))) for point in [[-1, -1, 1], [1, -1, 1]]]
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    lengths = np.linalg.norm(p[:, edges[:, 0]]-p[:, edges[:, 1]], axis=2)
    # Static collider is a faceted inscribed icosphere, so evaluate distance to
    # the actual collision mesh as well as analytic radius for honest reporting.
    collider, collider_faces = read_obj(out/'models/sphere.obj')
    normals = np.cross(collider[collider_faces[:, 1]]-collider[collider_faces[:, 0]], collider[collider_faces[:, 2]]-collider[collider_faces[:, 0]])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    offsets = np.sum(normals*collider[collider_faces[:, 0]], axis=1)
    inside_count = 0
    minimum_sphere_radius = float(np.linalg.norm(p, axis=2).min())
    for frame in p:
        # Sphere is convex and faces are outward-oriented.
        outside = np.max(frame @ normals.T-offsets, axis=1)
        inside_count += int((outside < -1e-4).sum())
    checks = {'finite': True, 'pinned_max_displacement_m': float(np.linalg.norm(p[:, pins]-p[0, pins], axis=2).max()),
        'minimum_sample_radius_m': minimum_sphere_radius, 'points_inside_faceted_collider': inside_count,
        'max_sampled_edge_stretch': float((lengths/lengths[:1]).max()),
        'sampled_edge_stretch_p99': float(np.percentile(lengths/lengths[:1], 99)),
        'sampled_edge_stretch_p999': float(np.percentile(lengths/lengths[:1], 99.9)),
        'min_z_m': float(p[:, :, 2].min()), 'max_abs_xy_m': float(np.abs(p[:, :, :2]).max())}
    checks['passed'] = (checks['pinned_max_displacement_m'] < 1e-5 and inside_count == 0
                        and checks['min_z_m'] > -1.201 and checks['max_sampled_edge_stretch'] < 1.2)
    source = Path(__file__).resolve().parent.parent/'upstream/BS-Cloth-headless'
    metadata = {'case': 'cloth_bspline', 'backend': 'BS-Cloth',
        'paper': 'Efficient B-Spline Finite Elements for Cloth Simulation, SIGGRAPH 2026',
        'source': 'https://github.com/Simulation-Intelligence/BS-Cloth',
        'source_commit': subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
        'device': 'cpu', 'geometry': 'mesh', 'frames': frames, 'fps': 24,
        'timestep_s': config['timestep'], 'substeps_per_output_frame': 5,
        'contact_resolution': config['bs_surfaces'][0]['linear_resolution'],
        'stretching_youngs_modulus_pa': config['stretching_youngs_modulus'],
        'contact_threshold_m': config['contact_threshold'],
        'configuration_audit': {'requested_bending_youngs_modulus': config['bending_youngs_modulus'],
            'serialized_bending_youngs_modulus': effective['bending_youngs_modulus'],
            'loader_fallback': config['bending_youngs_modulus'] != effective['bending_youngs_modulus'],
            'note': 'Read serialized initial state and log before treating requested material parameters as effective parameters.'},
        'control_points': 24*24, 'sampled_surface_vertices': len(first), 'sampled_faces': len(faces),
        'pinned_positions': first[pins].tolist(), 'checks': checks,
        'static_sphere': {'radius': .5, 'center': [0, 0, 0], 'collision_mesh': 'subdivision-2 icosphere'},
        'limitations': 'Preliminary single-resolution run. Dense surface evaluation adds no physical degrees of freedom; no material calibration or convergence study. Passed checks cover finite states, pins, sampled collider, ground and edge stretch <1.2 only, not all triangle intersections or physical accuracy.',
        'local_changes': 'Headless Linux build compatibility, thread limit, export-only evaluation of solved B-spline surface; solver equations unchanged.'}
    np.savez_compressed(out/'states.npz', positions=p, triangles=faces, t=np.arange(frames)/24)
    (out/'simulation.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    assert checks['passed'], checks


if __name__ == '__main__':
    main()
