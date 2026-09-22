"""Observation-to-question formatting; only model responses select commands."""
import math
from combat import WEAPON_NAMES

ACTIONS={k:k for k in ('attack','pickup','open_door','exit','retreat','explore','wait')}
WEAPON_DESCRIPTIONS={1:'Fists: melee, no ammunition needed.',2:'Pistol: use only when stronger guns have no ammo.',3:'Shotgun: prefer this powerful gun to the pistol when shells are available.',
                     4:'Chaingun: rapid bullet fire.',5:'Rocket launcher: powerful, dangerous at close range.',
                     6:'Plasma rifle: powerful rapid fire.',7:'BFG: powerful, consumes 40 cells per shot.'}


def request(s, memory, mission):
    enemies=s['enemies'][:3]
    failures=s.get('target_failures',{})
    items=sorted((i for i in memory.values() if str(i['id']) not in failures),key=lambda i:i['distance'])[:4]
    lines=[f"The player has {s['hp']:.0f} health and {s['armor']:.0f} armor.",
           'Inventory: '+ '; '.join(f"{WEAPON_NAMES[int(k)]} with {v['ammo']} ammo" for k,v in s['inventory'].items() if v['owned'])+'.']
    if enemies:
        lines.append('Hostile enemies in sight: '+ '; '.join(f"{e['name']} #{e['id']} at {e['distance']:.1f} meters" for e in enemies)+'.')
    else:lines.append('The area in sight is clear of enemies.')
    if items:lines.append('Observed ground items: '+ '; '.join(f"{i['name']} #{i['id']} ({i['category']}) at {i['distance']:.1f} meters" for i in items)+'.')
    if failures:lines.append('Unreachable pickup targets in the current area: '+', '.join('#'+key for key in failures)+'. Those commands are unavailable until the position or doors change.')
    commands={}
    criteria={}
    def add(key,description,action,target=None):
        criteria[key]=description
        commands[key]={'action':action,'target':target}
    for e in enemies:
        add(f"shoot_{e['id']}",f"Fight {e['name']} #{e['id']}, {e['distance']:.1f}m away.",'attack',dict(e))
    for i in items:
        add(f"collect_{i['id']}",f"Collect {i['name']} #{i['id']} ({i['category']}), {i['distance']:.1f}m away. {s.get('target_failures',{}).get(str(i['id']), '')}",'pickup',dict(i))
    d=s['door']
    if d:
        lines.append(f"A closed door is {d['distance']:.1f} meters away.")
        add('open_door','Approach and open the closed door.','open_door',dict(d))
    if mission and mission.exit:
        add('exit','No immediate threat or needed supplies: reach the exit to complete the level.','exit')
    if enemies:add('retreat','Back away from the enemies without firing.','retreat')
    add('explore','Search unknown areas when the current route is blocked.','explore')
    add('wait','Stay still without firing.','wait')
    execution=s.get('execution',{})
    if execution.get('status') in ('blocked','unavailable','arrived'):
        lines.append('Previous command result: '+execution.get('detail',execution['status'])+'.')
    weapons={WEAPON_NAMES[int(k)]:int(k) for k,v in s['inventory'].items() if v['owned']}
    nearest=min((e['distance'] for e in enemies),default=None)
    weapon_options={}
    for key,slot in weapons.items():
        ammo=s['inventory'][str(slot)]['ammo']
        loaded=slot==1 or ammo>=(40 if slot==7 else 1)
        reaches=nearest is not None and (slot!=1 or nearest<=1.5)
        kind='Melee only, range 1.5m' if slot==1 else ('Powerful ranged gun' if slot in (3,4,5,6,7) else 'Basic ranged gun')
        weapon_options[key]=f"{kind}. Loaded: {'yes' if loaded else 'no'}. Can hit visible enemies from here: {'yes' if loaded and reaches else 'no'}. Ammo: {ammo}."
    weapon_options['keep']='Keep the current weapon.'
    q={'command':{'type':'choice','instructions':'Finish the level alive. Fight visible threats, get needed supplies, open blocking doors, reach exit. Avoid unreachable targets and unnecessary detours.','criteria':criteria},
       'weapon':{'type':'choice','instructions':'Choose an effective weapon. Use a loaded gun for ranged enemies. Prefer shotgun to pistol when both are loaded. Use melee when there is no usable gun.','criteria':weapon_options}}
    return {'state':'\n'.join(lines),'questions':q,'commands':commands,'weapons':weapons}


def decode(result, packet, decision_id):
    chosen=result['answers']['command']['choice']
    weapon=result['answers']['weapon']['choice']
    return dict(packet['commands'][chosen],weapon=packet['weapons'].get(weapon),decision_id=decision_id,command=chosen)
