"""FreeDoom: Laya выбирает тактику, локальный контроллер исполняет её."""
import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

import numpy as np
import requests
import vizdoom as vzd

from doomlib.overlay import Overlay, Recorder
from doomlib.navigation import Navigator
from doomlib.combat import Combat, WEAPON_NAMES
from doomlib.items import Items
from doomlib.mission import Mission, map_data
from doomlib.report import build_report
from doomlib.executor import Executor
from doomlib.policy import request as policy_request, decode as policy_decode, ACTIONS

ROOT = Path(__file__).resolve().parent
TICRATE = 35
MARGIN = 0.15
MAX_HOLD = 105
DEPTH_M = 8.1 / 32
TACTICS = {
    'strafe_left_fire': 'Enemy visible: dodge left and shoot.',
    'strafe_right_fire': 'Enemy visible: dodge right and shoot.',
    'advance': 'Approach a distant enemy or clear passage.',
    'retreat': 'Escape nearby enemies when hurt or out of ammo.',
    'pickup': 'Collect visible health, ammunition or weapons.',
    'patrol': 'No enemies: explore corridors and find enemies.',
    'turn_around': 'Dead end: turn back to find another route.',
    'open_door': 'Closed door ahead: approach and open it.',
}
QUESTIONS = {
    'tactic': {
        'type': 'choice',
        'instructions': 'Choose the best next DOOM tactic to survive, kill enemies and explore.',
        'criteria': TACTICS,
    },
    'danger': {
        'type': 'noul',
        'instructions': 'Is the player in immediate danger from nearby enemies or low health?',
    },
}
BUTTONS = [vzd.Button.MOVE_FORWARD, vzd.Button.MOVE_BACKWARD,
           vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.TURN_LEFT_RIGHT_DELTA,
           vzd.Button.ATTACK, vzd.Button.USE] + [getattr(vzd.Button, f'SELECT_WEAPON{i}') for i in range(1,8)]


def wrap(angle):
    return (angle + 180) % 360 - 180


def bearing(x, y, px, py, angle):
    # Положительное значение — справа; положительный TURN также поворачивает вправо.
    return -wrap(math.degrees(math.atan2(y - py, x - px)) - angle)


def emit(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    handle.flush()


class Sensors:
    def __init__(self,door_sectors=()):
        self.door_sectors=set(door_sectors)
        self.visits = Counter()
        self.closed_doors = set()
        self.ever_opened = set()
        self.door_cooldowns = {}
        self.door_attempts = Counter()

    def read(self, game, tick):
        def var(name):
            return float(game.get_game_variable(getattr(vzd.GameVariable, name)))
        raw = game.get_state()
        px, py, angle = var('POSITION_X'), var('POSITION_Y'), var('ANGLE')
        self.visits[(round(px / 64), round(py / 64))] += 1
        depth = raw.depth_buffer
        def distance(lo, hi):
            band = depth[int(depth.shape[0] * .4):int(depth.shape[0] * .65), int(lo * depth.shape[1]):int(hi * depth.shape[1])]
            valid = band[band > 0]
            return round(float(np.percentile(valid, 25)) * DEPTH_M, 2) if valid.size else 0.0
        walls = {'left': distance(.06, .24), 'ahead': distance(.43, .57), 'right': distance(.76, .94)}
        rays = [{'bearing': round(math.degrees(math.atan(2*(col-.5))), 2),
                 'distance': distance(col-.015, col+.015)} for col in np.linspace(.05,.95,19)]
        enemies, items, obstacles, dead_ids = [], [], [], []
        for label in raw.labels:
            if str(label.object_category) == 'Self':
                continue
            d_units = math.hypot(label.object_position_x - px, label.object_position_y - py)
            obj = {'name': label.object_name, 'id': int(label.object_id),
                   'x': float(label.object_position_x), 'y': float(label.object_position_y), 'distance': round(d_units / 32, 2),
                   'bearing': round(bearing(label.object_position_x, label.object_position_y, px, py, angle), 1)}
            category = str(label.object_category)
            if category == 'Gore' or label.object_name.startswith('Dead'):
                dead_ids.append(int(label.object_id))
            if category == 'Monster' and not label.object_name.startswith('Dead'):
                ys, xs = np.where(raw.labels_buffer == label.value)
                if xs.size:
                    obj['aim_bearing'] = math.degrees(math.atan(2*(float(np.median(xs))+.5)/raw.screen_buffer.shape[1]-1))
                else:
                    obj['aim_bearing'] = obj['bearing']
                enemies.append(obj)
            elif category in ('Health', 'Armor', 'Ammo', 'Weapon', 'Key', 'Powerup'):
                obj['category'] = category
                items.append(obj)
            elif category in ('Hazard', 'LightSource', 'Obstacle'):
                obstacles.append(obj)
        enemies.sort(key=lambda o: o['distance'])
        items.sort(key=lambda o: o['distance'])
        closed = set()
        doors = []
        for i, sec in enumerate(raw.sectors):
            if i not in self.door_sectors:continue
            if sec.ceiling_height - sec.floor_height > 8 or not 2 <= len(sec.lines) <= 8:
                continue
            xs = [line.x1 for line in sec.lines] + [line.x2 for line in sec.lines]
            ys = [line.y1 for line in sec.lines] + [line.y2 for line in sec.lines]
            if max(xs) - min(xs) > 256 or max(ys) - min(ys) > 256:
                continue
            closed.add(i)
            # Целиться в ближайшую часть широкого проёма, а не в его далёкий центр.
            dx = float(np.clip(px,min(xs)+24,max(xs)-24)) if max(xs)-min(xs)>48 else (min(xs)+max(xs))/2
            dy = float(np.clip(py,min(ys)+24,max(ys)-24)) if max(ys)-min(ys)>48 else (min(ys)+max(ys))/2
            b = bearing(dx, dy, px, py, angle)
            dist = math.hypot(dx - px, dy - py) / 32
            if abs(b) < 43 and dist < 10 and self.door_cooldowns.get(i, -1) < tick:
                col = int(np.clip((.5 + math.tan(math.radians(b)) / 2) * depth.shape[1], 0, depth.shape[1] - 1))
                visible_depth = float(np.median(depth[int(depth.shape[0]*.4):int(depth.shape[0]*.6), max(0,col-2):col+3])) * DEPTH_M
                if dist <= visible_depth + 2.5:
                    doors.append({'id': i, 'x': dx, 'y': dy, 'opened_before': i in self.ever_opened,
                                  'distance': round(dist, 2), 'bearing': round(b, 1)})
        opened = self.closed_doors - closed
        self.closed_doors = closed
        self.ever_opened.update(opened)
        doors.sort(key=lambda d: d['distance'])
        candidates = []
        for side, offset in [('ahead', 0), ('left', 55), ('right', -55)]:
            a = math.radians(angle + offset)
            cell = (round((px + 128 * math.cos(a))/64), round((py + 128 * math.sin(a))/64))
            score = self.visits[cell] + (1000 if walls[side] < 1.4 else 0)
            candidates.append((score, -walls[side], side))
        least = min(candidates)[2]
        return raw, {
            'seconds': round(tick / TICRATE, 3), 'hp': var('HEALTH'), 'armor': var('ARMOR'),
            'ammo': var('SELECTED_WEAPON_AMMO'), 'weapon': int(var('SELECTED_WEAPON')),
            'kills': int(var('KILLCOUNT')), 'itemcount': int(var('ITEMCOUNT')),
            'hitcount': int(var('HITCOUNT')), 'damagecount': var('DAMAGECOUNT'),
            'x': px, 'y': py, 'angle': angle, 'engine_tic': game.get_episode_time(),
            'walls': walls, 'rays': rays, 'enemies': enemies, 'dead_ids': dead_ids,
            'inventory': {str(i): {'owned': int(var('WEAPON'+str(i))), 'ammo': int(var('AMMO'+str(i)))} for i in range(1,8)},
            'items': items, 'obstacles': obstacles, 'door': doors[0] if doors else None,
            'opened_doors': sorted(opened), 'least_explored': least,
        }


def state_text(s, tactic, held, history):
    enemies = '; '.join(f"{e['name']} {e['distance']:.1f}m bearing {e['bearing']:+.0f}" for e in s['enemies'][:3]) or 'none'
    items = '; '.join(f"{i['name']} {i['distance']:.1f}m bearing {i['bearing']:+.0f}" for i in s['items'][:3]) or 'none'
    w = s['walls']
    d = s['door']
    door = f"closed {d['distance']:.1f}m bearing {d['bearing']:+.0f}" if d else 'none visible'
    return (f"t={s['seconds']:.1f}s HP {s['hp']:.0f} armor {s['armor']:.0f} weapon {s['weapon']} ammo {s['ammo']:.0f} kills {s['kills']}\n"
            f"current tactic: {tactic}, held {held:.1f}s\n"
            f"enemies: {enemies}\nitems: {items}\n"
            f"walls: ahead {w['ahead']:.1f}m left {w['left']:.1f}m right {w['right']:.1f}m\n"
            f"door: {door}\nleast explored: {s['least_explored']}\n"
            f"recent tactics: {', '.join(history)}\nbearing: negative left, positive right")


class LayaClient:
    def __init__(self, endpoint, model):
        self.session = requests.Session()
        self.session.trust_env = False
        self.endpoint, self.model = endpoint, model

    def health(self):
        response = self.session.get(self.endpoint.rsplit('/', 1)[0] + '/health', timeout=5)
        response.raise_for_status()
        info = response.json()
        if info.get('status') != 'ok' or self.model not in info.get('models', {}):
            raise RuntimeError('Laya model is not ready')
        return info

    def predict(self, text, questions=None):
        questions=questions or QUESTIONS
        started=time.perf_counter()
        response=self.session.post(self.endpoint,json={'state':text,'questions':questions,'model':self.model},timeout=(3,15))
        response.raise_for_status()
        return self.validate(response.json(),started,questions)

    def validate(self,result,started,questions):
        raw={}
        for name,q in questions.items():
            a=result['answers'][name]
            if q['type']!='choice':continue
            p=a['probabilities'];total=sum(p.values())
            if set(p)!=set(q['criteria']) or any(not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for v in p.values()):
                raise ValueError('Invalid choice probabilities')
            tolerance=len(p)*.005+.0001 if self.model=='jev' else .002
            if total<=0 or abs(total-1)>tolerance:raise ValueError('Invalid probability sum')
            if a['choice'] not in p:raise ValueError('Invalid choice')
            raw[name]=dict(p)
            a['probabilities']={k:v/total for k,v in p.items()}
        if result.get('routing',{}).get('model')!=self.model:raise ValueError('Unexpected checkpoint')
        a=result['answers'].get('command',result['answers'].get('tactic'))
        return {'choice':a['choice'],'probabilities':a['probabilities'],'answers':result['answers'],
                'danger':result['answers'].get('danger',{}).get('noul',0),
                'latency_ms':round((time.perf_counter()-started)*1000,2),'server_latency_ms':result.get('latency_ms'),
                'tokens':result['usage']['input_tokens'],'routing':result['routing'],
                'cost_usd':result['usage'].get('cost',0),'request_id':result.get('id'),'raw_probabilities':raw,
                'choice_matches_argmax':a['probabilities'][a['choice']]>=max(a['probabilities'].values())-.0002}


class JevClient(LayaClient):
    def __init__(self):
        super().__init__('https://openrouter.ai/api/alpha/decisions','jev')
        key=os.environ.get('OPENROUTER_API_KEY')
        if not key:
            for line in (ROOT/'.env').read_text().splitlines():
                if line.startswith('OPENROUTER_API_KEY='):
                    key=line.split('=',1)[1].strip().strip('"').strip("'")
        if not key:raise RuntimeError('OPENROUTER_API_KEY is missing')
        self.session.headers['Authorization']='Bearer '+key

    def health(self):
        response=self.session.get('https://openrouter.ai/api/v1/key',timeout=(5,15))
        response.raise_for_status()
        data=response.json()['data']
        return {'status':'ok','models':{'jev':{'model':'typesafe/jev-1.13','provider':'OpenRouter'}},
                'limit_remaining':data.get('limit_remaining')}

    def predict(self,text,questions=None):
        questions=questions or QUESTIONS
        started=time.perf_counter()
        response=self.session.post(self.endpoint,json={'model':'typesafe/jev-1.13','state':text,'questions':questions},timeout=(5,15))
        response.raise_for_status()
        result=response.json()
        served=result.get('model','')
        if not served.startswith('typesafe/jev-1.13'):raise ValueError('Unexpected Jev checkpoint')
        result['routing']={'model':'jev','served_model':served,'provider':result.get('provider')}
        return self.validate(result,started,questions)


Controller=Executor


def make_game(args):
    game = vzd.DoomGame()
    game.set_doom_game_path(str(Path(vzd.__file__).parent / 'freedoom2.wad'))
    game.set_doom_map(args.map)
    game.set_doom_skill(args.skill)
    game.set_seed(args.seed)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)
    game.set_sectors_info_enabled(True)
    game.set_window_visible(args.show)
    game.set_sound_enabled(args.sound)
    game.set_render_hud(True)
    game.set_available_buttons(BUTTONS)
    game.set_episode_timeout(0)
    game.add_game_args('+freelook 0 +sv_cheats 1 +neverswitchonpickup 1')
    game.init()
    return game


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=300)
    parser.add_argument('--stop-after-level', action='store_true', help='Завершить после входа на следующий уровень и 3 секунд игры на нём')
    parser.add_argument('--map', default='MAP01')
    parser.add_argument('--skill', type=int, choices=range(1, 6), default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--record', nargs='?', const='auto')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--sound', action='store_true')
    parser.add_argument('--give', default='')
    parser.add_argument('--dry', action='store_true')
    parser.add_argument('--realtime', type=int, choices=[0, 1], default=1)
    parser.add_argument('--tag', default='')
    parser.add_argument('--model', choices=['english', 'multilingual', 'typed-decisions', 'doom-adapted', 'jev'], default='typed-decisions')
    parser.add_argument('--endpoint', default='http://127.0.0.1:8000/predict')
    parser.add_argument('--interval', type=float, default=.5, help='Минимальный интервал запросов, секунды')
    args = parser.parse_args()
    if args.model=='doom-adapted' and args.endpoint=='http://127.0.0.1:8000/predict':
        args.endpoint='http://127.0.0.1:8001/predict'
    if args.seconds <= 0 or args.interval <= 0:
        parser.error('--seconds and --interval must be positive')
    if not re.fullmatch(r'MAP\d{2}', args.map.upper()):
        parser.error('--map must be MAPxx')
    if args.give and not all(re.fullmatch(r'[a-zA-Z0-9_]+', v) for v in args.give.split(',')):
        parser.error('--give must contain comma-separated item names')
    tag = re.sub(r'[^\w-]', '_', args.tag)
    run = ROOT / 'runs' / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + ('_' + tag if tag else ''))
    run.mkdir(parents=True)
    client = JevClient() if args.model=='jev' else LayaClient(args.endpoint, args.model)
    health = {'status': 'dry', 'models': {}}
    # Health проверяется до старта игры: не превращаем ошибку сервера в эвристический прогон.
    if not args.dry:
        health = client.health()
    config = {'args': vars(args), 'wad_sha256':hashlib.sha256((Path(vzd.__file__).parent/'freedoom2.wad').read_bytes()).hexdigest(), 'protocol':'model-authority-v1', 'questions':'dynamic; exact request in decisions.jsonl', 'laya_health': health,
              'vizdoom': vzd.__version__, 'tics_per_second': TICRATE,
              'command_ttl_seconds':2, 'automatic_weapon_pickup_switch':False,
              'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                for name in ['agent.py', 'overlay.py', 'report.py', 'navigation.py', 'combat.py', 'items.py', 'mission.py', 'policy.py', 'executor.py']}}
    if health.get('laya_source_commit'):
        config['laya_source_commit'] = health['laya_source_commit']
    (run/'source').mkdir()
    for name in config['source_sha256']:(run/'source'/name).write_bytes((ROOT/name).read_bytes())
    (run / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2))
    print(f'RUN {run}', flush=True)
    handles = {name: (run / f'{name}.jsonl').open('x') for name in ['decisions', 'telemetry', 'events']}
    pool = ThreadPoolExecutor(max_workers=1)
    game = recorder = None
    future = None
    pending = None
    sensors, controller = Sensors(), Controller()
    stats = {'kills': 0, 'deaths': 0, 'decisions': 0, 'errors': 0, 'pickups': 0,
             'levels_completed': 0, 'doors_opened': 0, 'stuck_recoveries': 0, 'bumps': 0, 'damage': 0}
    counts, reflex_counts, resource_counts = Counter(), Counter(), Counter()
    latencies, tokens, costs = [], [], []
    decision = {}
    tactic = 'wait'
    history = deque([tactic], maxlen=3)
    switched, last_request = 0, -100000
    total_ticks = enemy_ticks = episode = 0
    previous = None
    current_map=args.map.upper()
    next_level_tick=None
    consecutive_errors = 0
    status = 'completed'
    started = time.perf_counter()
    fatal = None

    def event(name, tick, **extra):
        emit(handles['events'], {'event': name, 'game_seconds': round(tick/TICRATE, 3), **extra})

    def accept(result, tick, final=False):
        nonlocal tactic, switched, decision, consecutive_errors
        consecutive_errors = 0
        stats['decisions'] += 1
        latencies.append(result['latency_ms'])
        tokens.append(result['tokens'])
        costs.append(result.get('cost_usd',0))
        directive=policy_decode(result,pending['packet'],stats['decisions'])
        directive['expires_tick']=pending['tick']+2*TICRATE
        result['command']=result['choice']
        result['choice']=directive['action']
        probs=Counter()
        for key,p in result['probabilities'].items():probs[pending['packet']['commands'][key]['action']]+=p
        result['probabilities']=dict(probs)
        new=result['choice']
        stale=pending['episode']!=episode or tick-pending['tick']>2*TICRATE
        apply=not(final or stale)
        reason='run_finished' if final else ('stale' if stale else 'model_command')
        if apply:
            controller.accept(directive,tick)
            if new!=tactic:event('tactic',tick,previous=tactic,next=new,reason=reason)
            tactic,switched=new,tick
        result['directive']=directive
        row = {**result, **pending, 'game_seconds': round(tick/TICRATE, 3),
               'applied': apply, 'reason': reason, 'active_tactic': tactic}
        emit(handles['decisions'], row)
        decision = result
        if stats['decisions'] % 5 == 1 or apply:
            print(f"[{tick/TICRATE:6.1f}s] #{stats['decisions']:3} {result['latency_ms']:6.0f}ms {result['tokens']:4}tok -> {new} {'*' if apply else '-'} HP={pending['hp']:.0f}", flush=True)

    def api_error(exc, tick):
        nonlocal consecutive_errors, fatal
        consecutive_errors += 1
        stats['errors'] += 1
        # Не сохраняем response body, env или произвольный текст HTTP-ошибки.
        event('api_error', tick, error_type=type(exc).__name__, validation_message=str(exc) if isinstance(exc,ValueError) else None)
        print(f'Model API error: {type(exc).__name__}', flush=True)
        if consecutive_errors >= 3:
            fatal = 'Three consecutive model API errors'

    def spawn():
        for item in filter(None, args.give.split(',')):
            game.send_game_command('give ' + item)
        # Первые тики после старта движок игнорирует управление.
        game.make_action([0] * len(BUTTONS), 12)

    try:
        game = make_game(args)
        spawn()
        controller=Controller(game.get_state().sectors,Mission(map_data(game.get_doom_game_path(),current_map)))
        sensors=Sensors(controller.mission.data['door_sectors'])
        overlay = Overlay(ACTIONS, 'dry idle' if args.dry else args.model)
        if args.record:
            dest = run / 'video.mp4' if args.record == 'auto' else Path(args.record).expanduser().resolve()
            recorder = Recorder(dest)
        started = time.perf_counter()
        interval_ticks = max(1, round(args.interval * TICRATE))
        for tick in range(round(args.seconds * TICRATE)):
            if game.is_episode_finished():
                if game.is_player_dead():
                    stats['deaths'] += 1
                    event('death', tick)
                else:
                    stats['levels_completed']+=1
                    event('level_finished', tick, map=current_map, alive=True, engine_finished=True)
                    current_map=f'MAP{int(current_map[3:])+1:02d}'
                    game.set_doom_map(current_map)
                    next_level_tick=tick
                episode += 1
                game.new_episode()
                spawn()
                controller=Controller(game.get_state().sectors,Mission(map_data(game.get_doom_game_path(),current_map)))
                sensors=Sensors(controller.mission.data['door_sectors'])
                event('episode_started',tick,map=current_map,episode=episode,engine_map=game.get_doom_map(),x=float(game.get_game_variable(vzd.GameVariable.POSITION_X)),y=float(game.get_game_variable(vzd.GameVariable.POSITION_Y)))
                tactic, switched, previous = 'wait', tick, None
            raw, s = sensors.read(game, tick)
            s['map']=current_map
            if previous:
                for field, stat in [('kills', 'kills'), ('itemcount', 'pickups')]:
                    delta = max(0, s[field] - previous[field])
                    if delta:
                        stats[stat] += delta
                        event(stat, tick, count=delta)
                damage = max(0, previous['hp'] - s['hp'])
                if damage:
                    stats['damage'] += damage
                    event('damage', tick, amount=damage, hp=s['hp'])
            for door in s['opened_doors']:
                stats['doors_opened'] += 1
                event('door_opened', tick, sector=door)
            controller.observe(s,tick)
            if future is not None and future.done():
                try:
                    accept(future.result(), tick)
                except Exception as exc:
                    api_error(exc, tick)
                future = None
            if fatal:
                raise RuntimeError(fatal)
            if future is None and tick - last_request >= interval_ticks:
                packet=policy_request(s,controller.known,controller.mission)
                text=packet['state']
                pending={'tick':tick,'episode':episode,'state':text,'packet':packet,'snapshot_tactic':tactic,'hp':s['hp']}
                last_request=tick
                if args.dry:
                    result={'answers':{'command':{'choice':'wait'},'weapon':{'choice':'keep'}},
                            'choice':'wait','probabilities':{k:float(k=='wait') for k in packet['commands']},
                            'danger':0,'latency_ms':0,'tokens':0,'routing':{'model':'dry'}}
                    accept(result,tick)
                else:future=pool.submit(client.predict,text,packet['questions'])
            action,refs=controller.act(s,tick)
            tactic=s['execution']['action']
            gains=s.get('resource',{}).get('gains',{})
            if gains:
                resource_counts.update(gains)
                event('resource_gain',tick,gains=gains)
            for kind, stat in [('bump', 'bumps'), ('stuck', 'stuck_recoveries')]:
                if kind in refs:
                    stats[stat] += 1
                    event(kind, tick, walls=s['walls'])
            counts[tactic] += 1
            reflex_counts.update(refs)
            enemy_ticks += bool(s['enemies'])
            emit(handles['telemetry'], {**s, 'tick': tick, 'episode': episode, 'tactic': tactic,
                                       'buttons': action, 'reflexes': refs, 'pending': future is not None})
            if recorder:
                recorder.write(overlay.render(raw.screen_buffer, s, decision, tactic, stats, future is not None))
            elif tick == 0:
                overlay.render(raw.screen_buffer, s, decision, tactic, stats, future is not None).save(run / 'preview.png')
            game.make_action(action, 1)
            total_ticks = tick + 1
            previous = s
            if args.stop_after_level and next_level_tick is not None and tick-next_level_tick>=3*TICRATE:
                break
            if args.realtime:
                delay = started + total_ticks/TICRATE - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
        if previous:
            stats['kills'] += max(0, int(game.get_game_variable(vzd.GameVariable.KILLCOUNT)) - previous['kills'])
            stats['pickups'] += max(0, int(game.get_game_variable(vzd.GameVariable.ITEMCOUNT)) - previous['itemcount'])
            if game.is_player_dead():
                stats['deaths'] += 1
                event('death', total_ticks)
    except KeyboardInterrupt:
        status = 'interrupted'
    except Exception as exc:
        import traceback
        traceback.print_exc()  # <-- ajout temporaire pour debug
        fatal = type(exc).__name__ + ': ' + (str(exc) if isinstance(exc, RuntimeError) else 'see events')
        status = 'failed'
        event('fatal', total_ticks, error_type=type(exc).__name__)
    finally:
        game_wall_seconds = time.perf_counter() - started
        if future is not None:
            try:
                accept(future.result(timeout=20), total_ticks, final=True)
            except Exception as exc:
                api_error(exc, total_ticks)
        pool.shutdown(wait=True, cancel_futures=True)
        client.session.close()
        if game:
            game.close()
        if recorder:
            try:
                recorder.close()
            except Exception as exc:
                fatal, status = type(exc).__name__, 'failed'
        for handle in handles.values():
            handle.close()
        summary = {**stats, 'status': status, 'error': fatal, 'model': 'dry' if args.dry else args.model,
                   'final_map':current_map, 'game_seconds': round(total_ticks/TICRATE, 3), 'wall_seconds': round(game_wall_seconds, 3),
                   'latency_ms_median': round(statistics.median(latencies), 2) if latencies else None,
                   'latency_ms_p90': round(float(np.percentile(latencies, 90)), 2) if latencies else None,
                   'input_tokens_total': sum(tokens), 'input_tokens_avg': round(statistics.mean(tokens), 2) if tokens else 0,
                   'external_api_cost_usd': sum(costs), 'local_compute_cost_usd': None,
                   'enemy_in_view_share': round(enemy_ticks/max(1,total_ticks), 4),
                   'action_ticks': dict(counts), 'reflex_ticks': dict(reflex_counts),
                   'resource_gains': dict(resource_counts),
                   'video_frames': recorder.frames if recorder else 0,
                   'laya_health': health, 'run': str(run)}
        (run / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        print('REPORT', build_report(run), flush=True)
    return 1 if status == 'failed' or stats['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
