"""Dense DFSPH dam break with a reconstructed water mesh at every output time."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pysplishsplash as sph
import pysplashsurf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--radius', type=float, default=.0125)
    parser.add_argument('--frames', type=int, default=124)
    parser.add_argument('--substeps', type=int, default=24)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'states.npz').exists(), 'Use a fresh output directory'
    fps = 24
    dt = 1/fps/args.substeps
    box = Path(__file__).resolve().parent/'vendor/UnitBox.obj'
    assert box.is_file()
    scene = {
        'Configuration': {'particleRadius': args.radius, 'simulationMethod': 4,
            'gravitation': [0, -9.81, 0], 'cflMethod': 0, 'timeStepSize': dt,
            'boundaryHandlingMethod': 2, 'enableZSort': False,
            'stopAt': (args.frames-1)/fps-dt*.5},
        'Materials': [{'id': 'Fluid', 'density0': 1000, 'viscosityMethod': 1,
            'Standard viscosity': {'viscosity': .01}}],
        'RigidBodies': [{'geometryFile': str(box), 'translation': [0, .75, 0],
            'scale': [2.6, 1.6, 1.0], 'isDynamic': False, 'isWall': True,
            'mapInvert': True, 'mapResolution': [40, 40, 40]}],
        'FluidBlocks': [{'denseMode': 0, 'start': [-1.15, .1, -.32],
            'end': [-.45, 1.25, .32]}],
    }
    (out/'scene.json').write_text(json.dumps(scene, indent=2))
    base = sph.Exec.SimulatorBase()
    base.init(sceneFile=str(out/'scene.json'), useGui=False, initialPause=False,
              outputDir=str(out/'sph_cache'))
    base.initSimulation()
    sim = sph.Simulation.getCurrent()
    fluid = sim.getFluidModel(0)
    tm = sph.TimeManager.getCurrent()
    tm.setTimeStepSize(dt)
    count = fluid.numActiveParticles()
    surfaces = out/'surfaces'
    surfaces.mkdir(exist_ok=True)
    positions, times, records = [], [], []
    start = time.time()
    def capture(frame):
        target = frame/fps
        actual = tm.getTime()
        assert abs(actual-target) < 1e-6, (frame, actual, target)
        points = np.asarray([fluid.getPosition(i) for i in range(count)], dtype=np.float32)[:, [0, 2, 1]]
        assert np.isfinite(points).all()
        positions.append(points)
        times.append(actual)
        # Pipeline distance arguments are relative to particle radius. The raw
        # reconstruct_surface API instead takes absolute distances.
        mesh_data, reconstruction = pysplashsurf.reconstruction_pipeline(
            np.ascontiguousarray(points, dtype=np.float64), particle_radius=args.radius,
            rest_density=1000., smoothing_length=2., cube_size=.6,
            iso_surface_threshold=.6, mesh_smoothing_weights=True,
            mesh_smoothing_iters=8, mesh_cleanup=True, compute_normals=False,
            subdomain_grid=True, subdomain_num_cubes_per_dim=64)
        mesh = mesh_data.mesh
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        assert len(vertices) and len(faces) and np.isfinite(vertices).all()
        np.savez_compressed(surfaces/f'{frame:04d}.npz',
            vertices=vertices.astype(np.float32), faces=faces.astype(np.int32))
        records.append({'frame': frame, 'time': actual, 'vertices': len(vertices), 'faces': len(faces)})
        if frame % 6 == 0:
            print(f'frame={frame}/{args.frames} particles={count} vertices={len(vertices)} time={actual:.6f} elapsed={time.time()-start:.1f}s', flush=True)
    capture(0)
    step = 0
    def callback():
        nonlocal step
        step += 1
        # SPlisHSPlasH Real is float32. Reset its clock to the exact fixed-step
        # schedule rather than accumulating float32 timestamp drift.
        tm.setTime(step*dt)
        tm.setTimeStepSize(dt)
        if step % args.substeps == 0:
            capture(step//args.substeps)
    base.setTimeStepCB(callback)
    # This initializes boundaries, scripts and debug tools via deferredInit.
    # Calling timeStepNoGUI immediately after initSimulation skips that setup
    # in newer wheels and may dereference uninitialized debug-tool state.
    base.runSimulation()
    assert len(positions) == args.frames, len(positions)
    p = np.asarray(positions)
    np.savez_compressed(out/'states.npz', positions=p, t=np.asarray(times))
    bounds_min, bounds_max = p.min(axis=(0, 1)), p.max(axis=(0, 1))
    checks = {'finite': bool(np.isfinite(p).all()), 'particle_count': count,
        'bounds_min_xyz_m': bounds_min.tolist(), 'bounds_max_xyz_m': bounds_max.tolist(),
        'min_z_m': float(p[:, :, 2].min()), 'max_abs_x_m': float(np.abs(p[:, :, 0]).max()),
        'initial_com_x': float(p[0, :, 0].mean()), 'final_com_x': float(p[-1, :, 0].mean()),
        'max_timestamp_error_s': float(np.max(np.abs(np.asarray(times)-np.arange(args.frames)/fps)))}
    checks['inside_tank'] = bool(np.all(bounds_min >= np.array([-1.3, -.5, -.05])-.001)
                                 and np.all(bounds_max <= np.array([1.3, .5, 1.55])+.001))
    checks['passed'] = checks['finite'] and checks['inside_tank'] and checks['final_com_x'] > checks['initial_com_x']+.15
    metadata = {'case': 'liquid_dam_surface', 'backend': 'SPlisHSPlasH', 'solver': 'DFSPH',
        'device': 'cpu', 'geometry': 'reconstructed_surface_mesh', 'state_geometry': 'particles',
        'versions': {'pysplishsplash': '2.19.1', 'pysplashsurf': '0.14.1.0'},
        'radius': args.radius, 'fps': fps,
        'frames': args.frames, 'substeps_per_output_frame': args.substeps, 'timestep_s': dt,
        'tank': True, 'checks': checks, 'elapsed_seconds': time.time()-start,
        'surface_reconstruction': {'method': 'splashsurf SPH color field marching cubes',
            'smoothing_length_relative': 2., 'cube_size_relative': .6,
            'iso_surface_threshold': .6, 'weighted_smoothing_iterations': 8,
            'limitations': 'Surface visualization derived from solver particles; smoothing is not additional physical resolution.'},
        'limitations': 'Initial preview, not material calibration or convergence validation.'}
    (out/'simulation.json').write_text(json.dumps(metadata, indent=2))
    (out/'surface_manifest.json').write_text(json.dumps(records, indent=2))
    base.cleanup()
    assert checks['passed'], checks
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
