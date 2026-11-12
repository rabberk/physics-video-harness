"""Reconstruct a continuous isosurface from MPM sample occupancy, no dynamics edits."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;data=np.load(out/'states.npz');meta=json.loads((out/'simulation.json').read_text());target=out/'surfaces';target.mkdir(exist_ok=True)
spacing=.0125;sample_spacing=.025 if 'simulation_positions' in data else 2*meta['radius'];sigma=.6*sample_spacing/spacing;level=.2*(spacing/sample_spacing)**3;lower=np.array([-1.5,-1.5,-.12]);upper=np.array([1.5,1.5,1.2]);shape=np.ceil((upper-lower)/spacing).astype(int)
for i,q in enumerate(data['simulation_positions'] if 'simulation_positions' in data else data['positions']):
 idx=np.floor((q-lower)/spacing).astype(int);valid=np.all((idx>=0)&(idx<shape),axis=1);idx=idx[valid]
 field=np.zeros(shape,dtype=np.float32);np.add.at(field,tuple(idx.T),1);field=gaussian_filter(field,sigma)
 vertices,faces,_,_=marching_cubes(field,level=level,spacing=(spacing,)*3);vertices+=lower
 np.savez_compressed(target/f'{i:04d}.npz',vertices=vertices.astype('f'),faces=faces)
 if i%24==0:print('surface',i,len(vertices),flush=True)
meta['surface_reconstruction']={'method':'Gaussian occupancy isosurface','voxel_m':spacing,'sigma_voxels':sigma,'isovalue':level,'limitations':'Visualization approximation; not individual grains or a new physical solve.'};(out/'simulation.json').write_text(json.dumps(meta,indent=2))
