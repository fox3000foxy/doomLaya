"""Experiment integrity is separate from level-completion acceptance."""
import json,sys,math,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from check_authority import check
from check_navigation import analyze

def verify(run):
 run=Path(run);authority=check(run)
 rows=[json.loads(l) for l in (run/'telemetry.jsonl').read_text().splitlines()]
 events=[json.loads(l) for l in (run/'events.jsonl').read_text().splitlines()]
 summary=json.loads((run/'summary.json').read_text());errors=list(authority['errors'])
 if abs(summary['wall_seconds']-summary['game_seconds'])>max(2,summary['game_seconds']*.05):errors.append('Game did not sustain real time; RTT comparison is not valid')
 info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=width,height,nb_frames,r_frame_rate','-show_entries','format=duration','-of','json',str(run/'video.mp4')],text=True))
 stream=info['streams'][0]
 if (stream['width'],stream['height'],stream['r_frame_rate'])!=(1920,1080,'35/1'):errors.append('Video format mismatch')
 if int(stream['nb_frames'])!=len(rows):errors.append('Video frame mismatch')
 if not math.isclose(float(info['format']['duration']),len(rows)/35,abs_tol=.05):errors.append('Video duration mismatch')
 finish=[e for e in events if e['event']=='level_finished' and e.get('alive') and e.get('engine_finished')]
 next_spawn=[e for e in events if e['event']=='episode_started' and e.get('map')=='MAP02' and e.get('engine_map','').upper()=='MAP02']
 completed=bool(finish and next_spawn and sum(r['map']=='MAP02' for r in rows)>=105)
 navigation=analyze(rows)
 acceptance_errors=[] if completed else ['MAP01 was not completed and MAP02 was not played for three seconds']
 for field,limit in [('max_stationary_seconds',4),('max_no_new_cell_seconds',25),('max_no_new_region_seconds',25),('max_stationary_turn_reversals_per_second',5)]:
  if navigation[field]>limit:acceptance_errors.append(f'{field} exceeds {limit}')
 result={'experiment_valid':not errors,'passed':not errors and not acceptance_errors,'errors':errors,'level_completed':completed,
         'acceptance_errors':acceptance_errors,
         'authority':authority,'navigation':navigation,'video':info,'levels':finish,'summary':summary}
 (run/'verification.json').write_text(json.dumps(result,indent=2));return result
if __name__=='__main__':
 result=verify(sys.argv[1]);print(json.dumps(result,indent=2));raise SystemExit(not result['passed'])
