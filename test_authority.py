"""Regression checks for the model/controller authority boundary."""
import copy
import math
import json
from pathlib import Path
import unittest
from executor import Executor
from navigation import Navigator
from policy import request as policy_request
from mission import map_data
import vizdoom

SAMPLE=json.loads((Path(__file__).resolve().parent/'fixtures/authority-state.json').read_text())

class Motor:
    def observe(self,*args):pass
    def steer(self,*args,**kwargs):return [1,0,0,0,0,0,0],['waypoint']
    def state(self,*args):return {}
    def reject(self,*args):pass

class Exit:
    exit={'center':[0,0]}
    data={'name':'MAP01'}
    def objective(self):return (0,0)
    def activate(self,*args):return None

class AuthorityTest(unittest.TestCase):
    def setup_state(self):
        s=copy.deepcopy(SAMPLE);s['door']=None
        c=Executor(mission=Exit());c.navigator=Motor();c.observe(s,0)
        return c,s

    def test_failed_path_is_not_recomputed_every_frame(self):
        nav=Navigator();calls=[]
        nav.nearest=lambda xy:tuple(xy)
        nav.plan=lambda state,tick,goal:calls.append(tick)
        s={'x':0,'y':0,'angle':0}
        for tick in range(100):nav.steer(s,tick,(200,0))
        self.assertEqual(calls,[0,35,70])
        nav.steer(dict(s,x=32),100,(200,0))
        self.assertEqual(calls[-1],100)

    def test_moving_target_inside_one_grid_cell_does_not_trigger_replanning(self):
        nav=Navigator();calls=[]
        nav.nearest=lambda xy:(math.floor(xy[0]/16),math.floor(xy[1]/16))
        nav.plan=lambda state,tick,goal:calls.append(tick)
        for tick in range(100):nav.steer({'x':0,'y':0,'angle':0},tick,(200+tick*.01,0))
        self.assertEqual(calls,[0,35,70])

    def test_unknown_specific_destination_never_falls_back_to_exploration(self):
        nav=Navigator();nav.nearest=lambda xy:(0,0) if xy==(0,0) else None
        nav.plan({'x':0,'y':0},0,(1000,1000))
        self.assertEqual(nav.goal_kind,'unreachable')
        self.assertEqual(nav.path,[])

    def test_unexecutable_pickup_removed_without_selecting_another_action(self):
        c,s=self.setup_state();item=s['items'][0]
        s['target_failures']={str(item['id']):'Unreachable from the current area'}
        packet=policy_request(s,{item['id']:item},Exit())
        self.assertNotIn('collect_'+str(item['id']),packet['commands'])
        self.assertTrue({'exit','explore','wait'}<=set(packet['commands']))
        self.assertEqual(c.act(s,0)[0],[0]*14)

    def test_map01_wall_208_is_not_a_manual_door(self):
        data=map_data(Path(vizdoom.__file__).parent/'freedoom2.wad','MAP01')
        self.assertEqual(set(data['door_sectors']),{27,56,121,161})
        self.assertNotIn(208,data['door_sectors'])

    def test_weapon_animation_never_fires_the_previous_weapon(self):
        c,s=self.setup_state();enemy=dict(s['enemies'][0],aim_bearing=0,distance=1)
        s['enemies']=[enemy];s['weapon']=2;s['ammo']=20
        c.accept({'action':'attack','target':enemy,'weapon':1,'decision_id':8,'command':'shoot'},0)
        self.assertEqual(c.act(s,0)[0][5],0)
        self.assertEqual(c.act(s,1)[0][5],0)

    def test_idle_has_no_implicit_combat_pickup_or_weapon_switch(self):
        for command in (None,{'action':'wait','target':None,'weapon':None,'decision_id':1,'command':'wait'}):
            c,s=self.setup_state()
            if command:c.accept(command,0)
            buttons,_=c.act(s,0)
            self.assertEqual(buttons,[0]*14)
            self.assertIsNone(s['resource']['target'])

    def test_exit_with_monster_and_items_only_moves(self):
        c,s=self.setup_state();c.accept({'action':'exit','target':None,'weapon':None,'decision_id':2,'command':'exit'},0)
        buttons,_=c.act(s,0)
        self.assertEqual(buttons,[1]+[0]*13)
        self.assertIsNone(s['resource']['target'])

    def test_attack_exact_target_and_no_retarget_after_disappearance(self):
        c,s=self.setup_state();enemy=dict(s['enemies'][0],aim_bearing=0,distance=4)
        s['enemies'][0]=enemy;s['weapon']=2;s['ammo']=20
        c.accept({'action':'attack','target':enemy,'weapon':None,'decision_id':3,'command':'shoot'},0)
        self.assertEqual(c.act(s,0)[0][5],1)
        s['enemies']=[dict(enemy,id=enemy['id']+999)]
        self.assertEqual(c.act(s,1)[0],[0]*14)
        self.assertEqual(s['execution']['status'],'unavailable')

    def test_far_melee_approaches_selected_enemy_without_punching_air(self):
        c,s=self.setup_state();enemy=dict(s['enemies'][0],aim_bearing=0,distance=14)
        s['enemies']=[enemy];s['weapon']=1
        c.accept({'action':'attack','target':enemy,'weapon':1,'decision_id':7,'command':'shoot'},0)
        buttons,refs=c.act(s,0)
        self.assertEqual(buttons[0],1)
        self.assertEqual(buttons[5],0)
        self.assertIn('approach_selected_enemy',refs)
        self.assertEqual(s['execution']['target_id'],enemy['id'])

    def test_door_requires_its_own_command(self):
        c,s=self.setup_state();s['door']={'id':42,'x':s['x']+30,'y':s['y'],'distance':1,'bearing':0}
        c.accept({'action':'exit','target':None,'weapon':None,'decision_id':4,'command':'exit'},0)
        self.assertEqual(c.act(s,0)[0],[0]*14)
        self.assertEqual(s['execution']['status'],'blocked')

    def test_weapon_selection_is_explicit_and_expired_command_is_idle(self):
        c,s=self.setup_state();s['weapon']=2
        c.accept({'action':'wait','target':None,'weapon':1,'decision_id':5,'command':'wait','expires_tick':5},0)
        self.assertEqual(c.act(s,0)[0][7],1)
        self.assertEqual(c.act(s,6)[0],[0]*14)

    def test_pickup_is_exact_target_even_with_visible_enemies(self):
        c,s=self.setup_state();item=s['items'][0]
        c.accept({'action':'pickup','target':item,'weapon':None,'decision_id':6,'command':'collect'},0)
        buttons,_=c.act(s,0)
        self.assertEqual(s['resource']['target']['id'],item['id'])
        self.assertEqual(buttons[5:], [0]*9)

if __name__=='__main__':unittest.main()
