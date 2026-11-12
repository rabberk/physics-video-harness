"""Fixed MiniMax H3 Ref2VA inference over real solver-rendered draft videos."""
import argparse,json,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--draft',type=Path,required=True);p.add_argument('--prompt',required=True)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--model',default='/root/.cache/phyai/models/minimax-h3')
 p.add_argument('--steps',type=int,default=20);p.add_argument('--seed',type=int,default=0)
 p.add_argument('--height',type=int,default=384);p.add_argument('--width',type=int,default=640)
 p.add_argument('--device',default='cuda:0');p.add_argument('--min-free-gib',type=float,default=24)
 p.add_argument('--allow-short-reference',action='store_true',help='Diagnostic opt-in: allow a reference shorter than the fixed 124-frame output')
 p.add_argument('--appearance-edit',action='store_true',help='Explicitly replace clay appearance while preserving motion, shape and camera')
 p.add_argument('--caption-file',type=Path,help='Use an exact structured caption for controlled prompt comparisons')
 p.add_argument('--target-noise-reference-frames',type=int,choices=[56,73,124],help='Diagnostic: reproduce target noise drawn after a reference of this VAE frame count, so reference-length changes can be compared at fixed target noise')
 a=p.parse_args()
 if a.caption_file and a.appearance_edit:p.error('--caption-file and --appearance-edit are mutually exclusive')
 if a.caption_file and not a.caption_file.is_file():p.error('Caption file missing')
 if a.output.exists():p.error('Output exists; choose a new path')
 if not a.draft.is_file():p.error('Draft video missing')
 if not (Path(a.model)/'transformer_ref/diffusion_pytorch_model.safetensors.index.json').is_file():
  p.error('Ref2VA weights missing. FL2VA weights cannot substitute for transformer_ref.')
 import torch
 if not torch.cuda.is_available():p.error('CUDA unavailable')
 free,total=torch.cuda.mem_get_info(a.device)
 if free/2**30<a.min_free_gib:p.error(f'Only {free/2**30:.1f} GiB free; refusing to disturb existing GPU jobs')
 from diffusers import ModularPipeline,ComponentsManager
 from diffusers.hooks import apply_group_offloading
 from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3VideoReference
 from diffusers.utils import export_to_video
 reference=MiniMaxH3VideoReference.from_file(str(a.draft))
 import math
 reference_frames=len(reference.frames);reference_fps=float(reference.fps)
 normalized_frames=min(124,math.floor(reference_frames*24/reference_fps+0.5))
 if normalized_frames<124 and not a.allow_short_reference:
  p.error(f'Reference covers {normalized_frames} frames at 24 fps, shorter than the fixed 124-frame output. Render a full-duration simulation, or explicitly prepare a tail-held reference for diagnostics. Use --allow-short-reference only for intentional ablations.')
 reference_audit={'decoded_frames':reference_frames,'decoded_fps':reference_fps,'normalized_frames_at_24fps':normalized_frames,'expected_vae_frames':max(1,(normalized_frames-5)//17)*17+5,'target_frames':124,'target_fps':24,'short_reference_allowed':a.allow_short_reference}
 print('Reference timing: '+json.dumps(reference_audit),flush=True)
 manager=ComponentsManager()
 pipe=ModularPipeline.from_pretrained(a.model,workflow='ref2va',components_manager=manager,local_files_only=True)
 pipe.load_components(dtype=torch.bfloat16,pretrained_model_name_or_path=a.model,local_files_only=True)
 pipe.text_encoder.requires_grad_(False)
 apply_group_offloading(pipe.text_encoder.model,onload_device=torch.device(a.device),offload_device=torch.device('cpu'),
   offload_type='leaf_level',use_stream=True,non_blocking=True,low_cpu_mem_usage=True)
 transformer=pipe.transformer_ref
 transformer.requires_grad_(False)
 transformer.enable_group_offload(onload_device=torch.device(a.device),offload_device=torch.device('cpu'),
   offload_type='block_level',num_blocks_per_group=1,use_stream=True,non_blocking=True,low_cpu_mem_usage=True)
 def preserve_hooks(hooks,model_id,model,execution_device):return list(hooks)
 manager.enable_auto_cpu_offload(device=a.device,offload_strategy=preserve_hooks)
 prompt=('subject_definitions: <Video 1> is a physically simulated motion and geometry reference.\n'
  'summary: [reference generation] Generate a realistic video following <Video 1>.\n'
  'retention_analysis: <Video 1>: fully_preserved - preserve object count, spatial layout, motion direction, '
  'contact timing and event order; improve materials and lighting.\n'
  'detailed_description: '+a.prompt+'\noverall_soundscape: Natural quiet ambience.\nnon_diegetic_music: None.')
 if a.appearance_edit:
  prompt=('subject_definitions: <Video 1> is a grey clay simulation reference for object geometry, motion and camera.\n'
   'summary: [video editing] Turn the simulation draft in <Video 1> into live-action photographic footage of the described real materials and environment. The reference supplies motion and composition; the text supplies the finished appearance.\n'
   'retention_analysis: <Video 1>: partially_preserved - retain the main objects, starting positions, scale, spatial layout, motion paths, macroscopic deformation, contact timing, pinned constraints, event order and fixed camera. Recreate appearance as live-action photography: natural material microstructure, continuous surfaces, physically plausible reflections, soft contact shadows, detailed real-world textures and photographic lighting. Replace grey clay surfaces and flat synthetic lighting; coarse mesh facets, gaps and sample-particle markers are draft visualization artifacts, not target surface detail.\n'
   'detailed_description: '+a.prompt+'\noverall_soundscape: Natural quiet ambience.\nnon_diegetic_music: None.')
 if a.caption_file:
  prompt=a.caption_file.read_text().strip()
  if not prompt:p.error('Caption file is empty')
 noise_kwargs={}
 if a.target_noise_reference_frames is not None:
  from diffusers.modular_pipelines.minimax_h3.modular_pipeline import resolve_canvas_size
  reference_h,reference_w=resolve_canvas_size(reference.frames.shape[2],reference.frames.shape[1],32,768,768*1344)
  condition_t=((a.target_noise_reference_frames-5)//17)*5+2
  fixed_rng=torch.Generator('cpu').manual_seed(a.seed)
  torch.randn((1,24,condition_t,reference_h//16,reference_w//16),generator=fixed_rng,dtype=torch.float32)
  fixed_video=torch.randn((1,24,37,a.height//16,a.width//16),generator=fixed_rng,dtype=torch.float32)
  fixed_audio=torch.randn((414,32),generator=fixed_rng,dtype=torch.float32).reshape(2,207,32).permute(0,2,1)
  noise_kwargs={'latents':fixed_video,'audio_latents':fixed_audio}
  print('Fixed target noise: '+str(a.target_noise_reference_frames)+' reference frames; video sum '+str(fixed_video.sum().item()),flush=True)
 start=time.time()
 with torch.inference_mode():
  videos=pipe(prompt=prompt,references=[reference],
    num_frames=124,height=a.height,width=a.width,num_inference_steps=a.steps,
    generator=torch.Generator('cpu').manual_seed(a.seed),output='videos',**noise_kwargs)
 a.output.parent.mkdir(parents=True,exist_ok=True);export_to_video(videos[0],str(a.output),fps=24)
 a.output.with_suffix('.json').write_text(json.dumps({**vars(a),'draft':str(a.draft),'output':str(a.output),
   'caption_file':str(a.caption_file) if a.caption_file else None,
   'workflow':'ref2va','reference_timing':reference_audit,'full_prompt':prompt,'elapsed_seconds':time.time()-start,'trained_model':False},indent=2))
if __name__=='__main__':main()
