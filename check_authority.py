"""Audit every non-idle motor frame against an accepted, unexpired model response."""
import argparse,hashlib,json
from pathlib import Path

def check(run):
 run=Path(run);cfg=json.loads((run/'config.json').read_text());summary=json.loads((run/'summary.json').read_text())
 decisions=[json.loads(l) for l in (run/'decisions.jsonl').read_text().splitlines()]
 rows=[json.loads(l) for l in (run/'telemetry.jsonl').read_text().splitlines()]
 errors=[];accepted={}
 def require(condition,message):
  if not condition and len(errors)<30:errors.append(message)
 require(cfg.get('protocol')=='model-authority-v1','Wrong protocol')
 require(not cfg['args']['dry'],'Dry run has no model authority')
 require(not cfg['args']['give'],'Nonstandard inventory')
 require(summary['errors']==0,'API errors occurred')
 require(summary['status']=='completed','Run did not finish cleanly')
 for name,digest in cfg['source_sha256'].items():
  require(hashlib.sha256((run/'source'/name).read_bytes()).hexdigest()==digest,'Source snapshot changed: '+name)
 for d in decisions:
  p=d['packet'];directive=d['directive'];key=d['answers']['command']['choice']
  expected=p['commands'][key]
  require(directive['action']==expected['action'] and directive['target']==expected['target'],'Model target/action replaced')
  require(directive['weapon']==p['weapons'].get(d['answers']['weapon']['choice']),'Model weapon replaced')
  if d['applied']:accepted[directive['decision_id']]=d
 for index,s in enumerate(rows):
  a=s['buttons'];e=s['execution'];did=e['decision_id'];prefix=f"tick {s['tick']}: "
  require(s['tick']==index,prefix+'Missing tick')
  if did is None:
   require(not any(a),prefix+'Activity without an accepted decision');continue
  require(did in accepted,prefix+'Unknown decision ID')
  if did not in accepted:continue
  d=accepted[did];command=d['directive'];kind=command['action']
  require(d['episode']==s['episode'],prefix+'Decision from another episode')
  require(d['game_seconds']<=s['seconds']+.001,prefix+'Decision from future')
  require(s['tick']<=command['expires_tick'],prefix+'Expired decision')
  require(e['action']==kind,prefix+'Action override')
  require(e['target_id']==((command['target'] or {}).get('id')),prefix+'Target override')
  if a[5]:
   require(kind=='attack',prefix+'Unauthorized attack')
   if command['weapon'] is not None:require(s['weapon']==command['weapon'],prefix+'Firing a weapon other than the model requested')
   require(any(x['id']==e['target_id'] for x in s['enemies']),prefix+'Shooting at unobserved/different enemy')
  if a[6]:require(kind in ('open_door','exit'),prefix+'Unauthorized USE')
  for slot in range(1,8):
   if a[6+slot]:require(command['weapon']==slot,prefix+'Unauthorized weapon selection')
  if kind=='wait':require(not any(a[:7]),prefix+'Wait overridden')
 require(bool(accepted),'No accepted model decisions')
 require(summary['video_frames']==len(rows) if cfg['args']['record'] else True,'Video/telemetry frame mismatch')
 result={'pass':not errors,'frames':len(rows),'accepted_decisions':len(accepted),'errors':errors,
         'level_completed':summary['levels_completed']>0,'final_map':summary['final_map']}
 (run/'authority-verification.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('run');a=p.parse_args();r=check(a.run);print(json.dumps(r,indent=2));raise SystemExit(not r['pass'])
