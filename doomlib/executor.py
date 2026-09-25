"""Execute explicit model commands; no combat, item or weapon preference policy."""
import math
from doomlib.navigation import Navigator
from doomlib.combat import target_bearing


class Executor:
    def __init__(self,sectors=None,mission=None):
        self.navigator=Navigator(sectors)
        self.mission=mission
        self.directive=None
        self.known={}
        self.previous=None
        self.execution={'status':'waiting','detail':'No model decision yet'}
        self.anchor=None
        self.anchor_tick=0
        self.failures={}
        self.failure_origin=None
        self.escape_start=-1000
        self.escape_side=1
        self.last_selection=-1000

    def observe(self,s,tick):
        self.navigator.observe(s,tick)
        if self.failure_origin is not None and (math.dist((s['x'],s['y']),self.failure_origin)>64 or s.get('opened_doors')):
            self.failures.clear()
            self.failure_origin=None
        visible={i['id']:i for i in s['items']}
        self.known.update({k:dict(v) for k,v in visible.items()})
        for key,item in list(self.known.items()):
            item['distance']=math.hypot(item['x']-s['x'],item['y']-s['y'])/32
            if key not in visible and item['distance']<1.1:del self.known[key]
        gains={}
        if self.previous:
            for field in ('hp','armor'):
                diff=s[field]-self.previous[field]
                if diff>0:gains[field]=diff
            for slot in ('2','3','5','6'):
                diff=s['inventory'][slot]['ammo']-self.previous['inventory'][slot]['ammo']
                if diff>0:gains['ammo_'+slot]=diff
            for slot,entry in s['inventory'].items():
                diff=entry['owned']-self.previous['inventory'][slot]['owned']
                if diff>0:gains['weapon_'+slot]=diff
        self.previous={'hp':s['hp'],'armor':s['armor'],'inventory':s['inventory']}
        s['resource']={'gains':gains,'known_items':len(self.known),'target':None}
        s['execution']=dict(self.execution)
        s['target_failures']=dict(self.failures)

    def accept(self,directive,tick):
        old=self.directive or {}
        if (old.get('action'),(old.get('target') or {}).get('id')) != (directive['action'],(directive.get('target') or {}).get('id')):
            self.anchor=None
            self.anchor_tick=tick
            self.escape_start=-1000
        self.directive=directive

    def act(self,s,tick):
        d=self.directive or {'action':'wait','target':None,'weapon':None,'decision_id':None,'command':'wait'}
        if tick>d.get('expires_tick',float('inf')):
            d={'action':'wait','target':None,'weapon':None,'decision_id':None,'command':'expired'}
        kind,target=d['action'],d.get('target')
        a=[0.]*14
        refs=[]
        status='executing';detail=kind
        current=None
        if kind=='attack':
            current=next((e for e in s['enemies'] if e['id']==target['id']),None) if target else None
            if current:
                b=current.get('aim_bearing',current['bearing'])
                a[4]=max(-9,min(9,b))
                melee=s['weapon']==1
                usable=melee or s['ammo']>=(40 if s['weapon']==7 else 1)
                a[5]=float(abs(b)<5 and usable and (not melee or current['distance']<=1.5))
                refs=['aim_selected_enemy']
                if current['distance']>(1.4 if melee else 13):
                    move,motor_refs=self.navigator.steer(s,tick,(current['x'],current['y']))
                    if 'navigation_exhausted' not in motor_refs:
                        a[:5]=move[:5]
                        a[5]=float(a[5] and abs(move[4]-b)<5)
                        refs+=motor_refs+['approach_selected_enemy']
                    elif melee:
                        status,detail='blocked','Selected melee weapon is out of range and there is no path to the enemy'
            else:status,detail='unavailable','Selected enemy is no longer visible; select a new command'
        elif kind in ('pickup','open_door','exit','explore'):
            objective=None;available=True
            if kind=='pickup':
                current=self.known.get(target['id']) if target else None
                if current:
                    objective=(current['x'],current['y'])
                    s['resource']['target']=dict(current)
                else:available=False
            elif kind=='open_door':
                current=s['door'] if s['door'] and target and s['door']['id']==target['id'] else None
                if current:objective=(current['x'],current['y'])
                else:available=False
            elif kind=='exit':
                available=bool(self.mission and self.mission.exit)
                if available:objective=self.mission.objective()
            if not available:status,detail='unavailable','Selected target is no longer observable or present'
            else:
                a[:7],refs=self.navigator.steer(s,tick,objective)
                if kind=='pickup' and current['distance']<2:
                    b=target_bearing(current,s)
                    a[:7]=[float(abs(b)<20),0,0,0,max(-6,min(6,b)),0,0]
                    refs=['approach_selected_item']
                if kind=='open_door' and current['distance']<2.8:
                    b=target_bearing(current,s)
                    a[:7]=[float(abs(b)<20 and current['distance']>1.8),0,0,0,max(-6,min(6,b)),0,float(abs(b)<12 and tick%12==0)]
                    refs=['open_selected_door']
                if kind=='exit':
                    activation=self.mission.activate(s,tick)
                    if activation is not None:a[:7],refs=activation,['activate_selected_exit']
                door=s['door']
                if kind!='open_door' and 'activate_selected_exit' not in refs and door and door['distance']<2.7 and abs(door['bearing'])<25 and a[0]:
                    a[:7]=[0.]*7
                    status,detail='blocked','A closed door blocks movement; choose open_door to open it'
                elif 'navigation_exhausted' in refs:
                    status,detail='blocked','No navigable path to the selected target'
                    if target:
                        self.failures[str(target['id'])]='Unreachable from the current area'
                        if self.failure_origin is None:self.failure_origin=(s['x'],s['y'])
                elif target:self.failures.pop(str(target['id']),None)
        elif kind=='retreat':a[1]=1
        else:status,detail='waiting','Model chose wait' if d['decision_id'] is not None else 'No accepted model command'
        # Collision recovery changes motor commands, never the requested objective.
        xy=(s['x'],s['y'])
        moving=bool(any(a[:4]))
        if not moving or self.anchor is None or math.dist(xy,self.anchor)>24:
            self.anchor,self.anchor_tick=xy,tick
        if moving and tick-self.anchor_tick>=35 and tick-self.escape_start>=70:
            self.escape_start=tick
            self.escape_side=-1 if s['walls']['left']>s['walls']['right'] else 1
            self.navigator.reject(tick)
            refs.append('stuck')
        elapsed=tick-self.escape_start
        if kind in ('attack','exit','explore','pickup','open_door','retreat') and status=='executing' and elapsed<42:
            a[:5]=[0,float(elapsed<18),0,0,self.escape_side*7 if elapsed>=18 else 0]
            a[5]=0
            a[6]=0
            refs.append('recover_same_goal')
        selection=d.get('weapon')
        if selection and s['weapon']!=selection:a[5]=0
        if selection and s['inventory'][str(selection)]['owned'] and s['weapon']!=selection and tick-self.last_selection>=12:
            a[6+selection]=1
            a[5]=0
            self.last_selection=tick
            refs.append('select_model_weapon')
        if target:detail+=f" ({target.get('name',kind)} #{target['id']})"
        self.execution={'status':status,'detail':detail,'decision_id':d['decision_id'],'command':d['command'],'action':kind,
                        'target_id':target['id'] if target else None,'weapon':selection}
        s['execution']=dict(self.execution)
        s['combat']={'mode':kind,'target_id':target['id'] if kind=='attack' and target else None,
                     'target_name':target['name'] if target and 'name' in target else None,
                     'target_visible':current is not None,'preferred_weapon':selection,'selection':next((i for i in range(1,8) if a[6+i]),0)}
        s['mission']={'map':self.mission.data['name'],'exit':self.mission.exit} if self.mission else None
        s['navigation']=self.navigator.state(tick)
        return a,refs
