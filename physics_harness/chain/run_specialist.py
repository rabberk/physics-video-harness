"""Single entry point: orchestrate specialist solvers using one Newton trajectory."""
import argparse,os,subprocess,json,time,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('outputs/chain_specialist_20261111'));p.add_argument('--device',default='cuda:0');a=p.parse_args();root=Path(__file__).resolve().parents[2];out=a.output.resolve();scripts=root/'physics_harness/chain';py=sys.executable;started=time.time();out.mkdir(parents=True,exist_ok=True)
def run(script,*args,env=None):subprocess.run([py,str(scripts/script),*map(str,args)],env=env,check=True,cwd=root)
if not (out/'rigid.json').exists():run('solve_newton.py','--output',out,'--device',a.device)
jobs=[]
def start(name,cmd,cwd=root,env=None):
 log=open(out/f'{name}.log','w');jobs.append((name,subprocess.Popen(list(map(str,cmd)),cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT)))
if not (out/'sand/simulation.json').exists():start('sand',[py,scripts/'reconstruct_mpm.py','--output',out])
if not (out/'liquid/simulation.json').exists():
 env=dict(os.environ,LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0',OMP_NUM_THREADS='12',OPENBLAS_NUM_THREADS='1');start('liquid',[py,scripts/'solve_dfsph.py','--output',out],env=env)
if not (out/'cloth/simulation.json').exists():
 run('prepare_bscloth.py','--output',out);env=dict(os.environ,PHYAI_CPU_THREADS='12',PHYAI_EXPORT_PREVIEW='1',PHYAI_EXPORT_STRIDE='5',OMP_NUM_THREADS='12');start('cloth',[root/'upstream/BS-Cloth-headless/bin/Release-linux-x86_64/B-Spline-IPC/B-Spline-IPC'],cwd=out/'cloth',env=env)
if not (out/'smoke/simulation.json').exists():start('smoke',['xvfb-run','-a','blender','-b','-t','6','--python',scripts/'bake_interactions.py','--',out,'smoke'])
for name,job in jobs:
 code=job.wait()
 # Blender 3.0 may crash during fluid shutdown after successful cache save.
 if code and not (name=='smoke' and (out/'smoke/simulation.json').exists()):raise RuntimeError((name,code))
if not (out/'cloth/simulation.json').exists():run('export_chain_cloth.py','--output',out)
(out/'harness.json').write_text(json.dumps({'solvers':{'rigid_sand':'Newton XPBD + implicit MPM','cloth':'BS-Cloth B-spline FEM + IPC','liquid':'SPlisHSPlasH DFSPH + splashsurf','smoke':'Mantaflow'},'clock':{'fps':24,'frames':124},'elapsed_seconds':time.time()-started,'coupling':'Newton trajectory distributed to specialist adapters; only rigid-MPM impulse exchange is bidirectional.'},indent=2));print('SPECIALIST SOLVERS COMPLETE',flush=True)
