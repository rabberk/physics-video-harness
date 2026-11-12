"""Continuous display surface of Newton MPM states, no changes to dynamics."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--sigma',type=float,default=.65);p.add_argument('--level',type=float,default=.18);a=p.parse_args();root=a.output.resolve();data=np.load(root/'rigid.npz');meta=json.loads((root/'rigid.json').read_text());out=root/'sand/surfaces';out.mkdir(parents=True,exist_ok=True)
h=.008;dx=meta['particle_spacing'];sigma=a.sigma*dx/h;level=a.level*(h/dx)**3
for i,q in enumerate(data['sand']):
 lower=np.floor((q.min(axis=0)-.05)/h)*h;upper=q.max(axis=0)+.05;shape=np.ceil((upper-lower)/h).astype(int);indices=np.floor((q-lower)/h).astype(int);field=np.zeros(shape,dtype='f');np.add.at(field,tuple(indices.T),1);field=gaussian_filter(field,sigma)
 vertices,faces,_,_=marching_cubes(field,level=level,spacing=(h,)*3);vertices+=lower;np.savez_compressed(out/f'{i:04d}.npz',vertices=vertices.astype('f'),faces=faces.astype('i'))
 if i%12==0:print('sand surface',i,len(vertices),flush=True)
(root/'sand/simulation.json').write_text(json.dumps({'backend':'Newton implicit MPM states / occupancy isosurface visualization','frames':len(data['t']),'fps':24,'spacing_m':h,'sigma_voxels':sigma,'level':level,'note':'Continuous display surface from physical MPM states. Material samples are not DEM grains; surface reconstruction adds no physics.'},indent=2))
