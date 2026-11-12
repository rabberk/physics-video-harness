import subprocess,time,json
from pathlib import Path
root=Path(__file__).resolve().parent;out=root.parent/'outputs/physics_harness_v1/newton_rolling_sand_dense'
while not (out/'simulation.json').exists():time.sleep(2)
m=json.loads((out/'simulation.json').read_text())
if not m['checks']['passed']:raise RuntimeError('Simulation checks failed')
with (out/'render.log').open('w') as f:subprocess.run(['xvfb-run','-a','blender','-b','-t','8','--python-exit-code','1','--python',str(root/'render.py'),'--',str(out)],stdout=f,stderr=subprocess.STDOUT,check=True)
subprocess.run(['ffmpeg','-y','-v','error','-framerate',str(m['fps']),'-i',str(out/'frames/%04d.png'),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(out/'draft.mp4')],check=True)
subprocess.run(['ffmpeg','-v','error','-i',str(out/'draft.mp4'),'-f','null','-'],check=True)
print('draft complete',flush=True)
