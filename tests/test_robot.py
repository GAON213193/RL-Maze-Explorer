import unittest
from environment.maze_generator import Maze,generate_maze
from environment.robot import *

def corridor():
    g=[[1]*7 for _ in range(7)]
    for y in range(1,7):g[y][3]=0
    return Maze(g,(3,6),(3,1))

class RobotTests(unittest.TestCase):
    def env(self,**kwargs):
        e=RobotEnv(Config(**kwargs));e.reset(maze=corridor());return e
    def test_rotation_and_costs(self):
        e=self.env();start=e.battery
        e.step(LEFT);self.assertEqual(e.direction,3);self.assertEqual(e.battery,start-.5)
        e.step(RIGHT);e.step(FORWARD)
        self.assertEqual(e.position,(3,5));self.assertEqual(e.battery,start-2)
        self.assertEqual(e.turns,3)
    def test_collision(self):
        e=self.env();e.step(LEFT);pos=e.position;b=e.battery
        o,_,_,_,_=e.step(FORWARD)
        self.assertEqual(e.position,pos);self.assertTrue(o['collision'])
        self.assertEqual(e.battery,b-1);self.assertEqual(e.collisions,1)
    def test_charge_preserves_memory(self):
        e=self.env(view_distance=2);e.step(FORWARD);e.step(RIGHT);e.step(RIGHT);e.step(FORWARD)
        memory=e.observation()['memory'];t=e.turns
        e.step(CHARGE)
        self.assertEqual(e.battery,e.capacity);self.assertEqual(e.charges,1)
        self.assertEqual(e.turns,t+5);self.assertEqual(memory,e.observation()['memory'])
    def test_charge_elsewhere(self):
        e=self.env();e.step(FORWARD);b=e.battery;e.step(CHARGE)
        self.assertEqual(e.battery,b);self.assertEqual(e.charges,0)
    def test_charging_time_limit(self):
        e=self.env(max_turns=3);e.step(LEFT);b=e.battery;e.step(CHARGE)
        self.assertEqual(e.battery,b);self.assertTrue(e.truncated);self.assertEqual(e.charges,0)
        self.assertEqual(e.turns,3)
    def test_no_hidden_goal_and_observation_copy(self):
        e=self.env(view_distance=2);o=e.observation()
        self.assertNotIn('goal',o);self.assertNotIn('grid',o);self.assertNotIn('oracle_energy',o)
        self.assertEqual(o['memory'][1][3],UNKNOWN)
        o['memory'][1][3]=GOAL;self.assertEqual(e.memory[1][3],UNKNOWN)
    def test_wall_occlusion(self):
        e=self.env();e.maze.grid[4][3]=1;e._observe()
        self.assertIn((3,4),e.visible);self.assertNotIn((3,3),e.visible)
    def test_corner_occlusion(self):
        m=Maze([[0]*5 for _ in range(5)],(2,2),(4,4));m.grid[2][3]=1
        self.assertFalse(line_visible(m,(2,2),(3,3)))
    def test_turn_changes_fov(self):
        e=self.env(view_distance=2);self.assertIn((3,5),e.visible)
        e.step(RIGHT);e.step(RIGHT);self.assertNotIn((3,5),e.visible)
        self.assertEqual(e.memory[5][3],OPEN)
    def test_depletion_and_reset(self):
        e=self.env(battery_margin=1);e.step(FORWARD);e.energy_units=0
        _,_,done,_,_=e.step(LEFT);self.assertTrue(done)
        with self.assertRaises(RuntimeError):e.step(RIGHT)
        e.reset(maze=corridor());self.assertFalse(e.terminated);self.assertEqual(e.turns,0)
    def test_zero_energy_station_can_charge(self):
        e=self.env();e.energy_units=0;e.step(CHARGE)
        self.assertEqual(e.battery,e.capacity);self.assertFalse(e.terminated)
    def test_oracle_includes_turns_and_success_at_zero(self):
        m=corridor();turns,energy,actions=minimum_turn_route(m,1)
        self.assertEqual((turns,energy),(6,5.5))
        e=self.env(battery_margin=1);e.reset(maze=m,direction=1)
        for a in actions:o,r,done,trunc,info=e.step(a)
        self.assertTrue(done);self.assertEqual(e.reason,'success');self.assertEqual(e.battery,0)
        self.assertEqual(o['memory'][1][3],GOAL)
    def test_generated_routes_really_fit_budget(self):
        for algo in ('dfs','prim'):
            e=RobotEnv(Config(size=16,algorithm=algo,battery_margin=1));e.reset(seed=11)
            turns,energy,actions=minimum_turn_route(e.maze,e.direction)
            for a in actions:e.step(a)
            self.assertEqual(e.reason,'success');self.assertEqual(e.turns,turns);self.assertEqual(e.battery,0)
    def test_invalid_configuration_and_action(self):
        for kwargs in ({'battery_margin':.5},{'view_distance':0},{'charge_turns':0},{'collision_cost':.3}):
            with self.assertRaises(ValueError):Config(**kwargs)
        e=self.env()
        for a in (-1,4,True,'left'):
            with self.assertRaises(ValueError):e.step(a)
