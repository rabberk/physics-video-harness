"""Decode a generated video and produce time-aligned visual review artifacts."""
import argparse,json,subprocess
from pathlib import Path
import imageio.v3 as iio
import numpy as np
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('--case-dir',type=Path,required=True);p.add_argument('--seed',type=int,default=0);a=p.parse_args()
root=a.case_dir;final=root/f'final_seed{a.seed}.mp4';draft=root/'draft.mp4'
subprocess.run(['ffmpeg','-v','error','-i',str(final),'-f','null','-'],check=True)
info=[]
for path in (draft,final):
 frames=list(iio.imiter(path,plugin='pyav'))
 info.append({'path':str(path),'frames':len(frames),'shape':list(frames[0].shape),'finite':bool(all(np.isfinite(f).all() for f in frames))})
 if len(frames)!=124:raise ValueError(f'{path}: expected 124 frames, got {len(frames)}')
 if path==draft:original=frames
 else:generated=frames
subprocess.run(['ffmpeg','-y','-v','error','-i',str(draft),'-i',str(final),'-filter_complex','[0:v][1:v]hstack=inputs=2[v]','-map','[v]','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(root/f'comparison_seed{a.seed}.mp4')],check=True)
indices=[0,8,16,24,48,96];w,h=640,384;sheet=Image.new('RGB',(w*2,(h+28)*len(indices)),(245,245,245));draw=ImageDraw.Draw(sheet)
for row,i in enumerate(indices):
 y=row*(h+28);draw.text((10,y+6),f't={i/24:.3f}s | simulation draft (left) / H3 Ref2VA (right)',fill=(0,0,0))
 sheet.paste(Image.fromarray(original[i]).resize((w,h)),(0,y+28));sheet.paste(Image.fromarray(generated[i]).resize((w,h)),(w,y+28))
sheet.save(root/f'contact_sheet_seed{a.seed}.jpg')
(root/f'validation_seed{a.seed}.json').write_text(json.dumps({'videos':info,'note':'Decode, frame count and pixel checks only. Physical fidelity requires separate review.'},indent=2))
print(json.dumps(info,indent=2))
