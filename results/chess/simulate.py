from pathlib import Path
import mujoco,numpy as np,json
p=Path(__file__).parent
xml=['<mujoco><option timestep="0.001" gravity="0 0 -9.81" iterations="100" cone="elliptic"/><default><geom friction=".5 .002 .0002" solref=".008 1"/></default><worldbody><geom name="board" type="box" pos="0 0 -.01" size=".18 .18 .01"/>'];names=[]
for side in range(2):
 for rank in range(2):
  for c in range(8):
   name=f'p_{side}_{rank}_{c}';names.append(name);x=(c-3.5)*.04;y=(-3.5+rank)*.04 if side==0 else (3.5-rank)*.04
   if (side,rank,c)==(1,1,4):y=-.012
   h=.045 if rank else [.055,.060,.065,.075,.080,.065,.060,.055][c];moving=(side,rank,c)==(0,1,4)
   xml.append(f'<body name="{name}" pos="{x} {y} .001"'+'><freejoint/>')
   xml.append(f'<geom name="{name}_base" type="cylinder" pos="0 0 .004" size=".012 .004" mass=".014"/><geom name="{name}_stem" type="capsule" fromto="0 0 .008 0 0 {h-.009}" size=".005" mass=".004"/><geom name="{name}_head" type="sphere" pos="0 0 {h-.007}" size=".008" mass=".004"/></body>')
xml.append('<body name="held_target" mocap="true" pos=".02 -.10 .001"/></worldbody><equality><weld name="grip" body1="held_target" body2="p_0_1_4" solref=".004 1" solimp=".85 .95 .001"/></equality></mujoco>');(p/'scene.xml').write_text(''.join(xml));m=mujoco.MjModel.from_xml_string(''.join(xml));d=mujoco.MjData(m)
def path(t):
 knots=[(0,[.02,-.10,.001]),(.55,[.02,-.10,.001]),(1.15,[.02,-.10,.027]),(1.8,[.02,-.045,.012]),(2.15,[.02,-.020,.027]),(2.40,[.02,.008,.027]),(2.8,[.02,-.038,.020]),(3.4,[.02,-.038,.001]),(5.2,[.02,-.038,.001])]
 for (a,x),(b,y) in zip(knots,knots[1:]):
  if t<=b:u=np.clip((t-a)/(b-a),0,1);u=u*u*(3-2*u);return np.array(x)*(1-u)+np.array(y)*u
 return np.array(knots[-1][1])
frames=[];times=[];events=[];tilts=[];hid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'p_1_1_4');next_frame=0
for step in range(5126):
 t=step*.001;d.mocap_pos[0]=path(t)
 if t>=3.6:d.eq_active[0]=0
 mujoco.mj_step(m,d)
 if t+1e-9>=next_frame/24:
  frame=[]
  for name in names:
   bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,name);frame.extend(d.xpos[bid]);frame.extend(d.xquat[bid])
  frames.append(frame);times.append(t);next_frame+=1;tilts.append(float(np.rad2deg(np.arccos(np.clip(d.xmat[hid].reshape(3,3)[2,2],-1,1)))))
 for i in range(d.ncon):
  c=d.contact[i];a=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom1);b=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom2)
  if 'p_0_1_4' in a+b and 'p_1_1_4' in a+b and step%10==0:events.append(t)
np.savez_compressed(p/'states.npz',qpos=np.array(frames),times=times)
report={'solver':'MuJoCo','white_motion':'dynamic pawn held by finite-compliance weld to external target, released at 3.6s; hand omitted','black_motion':'free rigid body, contact-driven','frames':len(frames),'fps':24,'first_white_black_contact_s':min(events) if events else None,'black_max_tilt_deg':max(tilts),'black_final_tilt_deg':tilts[-1]};(p/'physics.json').write_text(json.dumps(report,indent=2));print(report)
(p/'render_data.json').write_text(json.dumps({'qpos':frames}))
