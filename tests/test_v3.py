import unittest
import numpy as np
from environment.robot import Config, RIGHT, LEFT, CHARGE, GOAL, OPEN, UNKNOWN
from environment.maze_generator import Maze
from environment.robot_v3 import RangeRobotEnv
from environment.gym_v3 import RangeChargeEnv, energy_route


def corridor():
    grid=[[1]*17 for _ in range(17)]
    for x in range(1,16): grid[1][x]=0
    grid[2][4]=grid[2][5]=0
    return Maze(grid,(1,1),(15,1))

class V3Tests(unittest.TestCase):
    def test_near_junction_far_axis_only(self):
        robot=RangeRobotEnv(Config(size=17,charge_turns=2))
        obs=robot.reset(maze=corridor(),direction=1)
        self.assertEqual(obs['forward_range'],14)
        self.assertEqual(obs['memory'][2][4],OPEN)  # forward 3
        self.assertEqual(obs['memory'][2][5],UNKNOWN)  # forward 4
        self.assertEqual(obs['memory'][1][15],OPEN)  # range does not identify far goal
        robot.position=(2,1);robot._observe()
        self.assertEqual(robot.observation()['memory'][2][5],OPEN)
        robot.position=(1,1);robot._observe()
        self.assertEqual(robot.observation()['memory'][2][5],OPEN)  # remembered
        self.assertNotIn((5,2),robot.visible)
        robot.position=(12,1);robot._observe()
        self.assertEqual(robot.observation()['memory'][1][15],GOAL)

    def test_wall_blocks_all_detail_beyond_it(self):
        maze=corridor();maze.grid[1][8]=1
        # Reachable alternate route for reset oracle, not in the viewing ray.
        for x in range(1,16): maze.grid[3][x]=0
        maze.grid[2][1]=maze.grid[2][15]=0
        robot=RangeRobotEnv(Config(size=17,charge_turns=2))
        obs=robot.reset(maze=maze,direction=1)
        self.assertEqual(obs['forward_range'],6)
        self.assertEqual(obs['memory'][1][9],UNKNOWN)

    def test_charge_two_turns_and_memory_retained(self):
        env=RangeChargeEnv(Config(size=7,charge_turns=2),0)
        env.reset(seed=8)
        env.step(RIGHT)
        before=env.robot.observation()['memory']; turns=env.robot.turns
        _,reward,_,_,info=env.step(CHARGE)
        self.assertEqual(env.robot.turns-turns,2)
        self.assertEqual(env.robot.battery,env.robot.capacity)
        self.assertEqual(before,env.robot.observation()['memory'])
        self.assertEqual(info['charges'],1)
        self.assertLessEqual(reward,0)  # no charge farming reward

    def test_charge_threshold_boundaries_and_location(self):
        for units, allowed in [(11,False),(10,True),(9,True)]:
            with self.subTest(units=units):
                env=RangeChargeEnv(Config(size=7,charge_turns=2,charge_threshold=0.5),0)
                env.reset(seed=8)
                env.robot.capacity_units=20
                env.robot.energy_units=units
                _,_,_,_,info=env.step(CHARGE)
                self.assertEqual(env.robot.energy_units,20 if allowed else units)
                self.assertEqual(info['charge_completed'],allowed)
                self.assertEqual(info['charges'],int(allowed))
                self.assertEqual(env.robot.turns,2 if allowed else 1)
        env.reset(seed=8)
        env.robot.position=next((x,y) for y,row in enumerate(env.robot.maze.grid)
                                for x,v in enumerate(row) if v==0 and (x,y) not in
                                (env.robot.maze.start,env.robot.maze.goal))
        env.robot.energy_units=1
        _,_,_,_,info=env.step(CHARGE)
        self.assertFalse(info['charge_completed'])
        self.assertEqual(env.robot.energy_units,1)

    def test_64_defaults_and_valid_observation(self):
        env=RangeChargeEnv()
        self.assertEqual(env.config.size,64)
        self.assertEqual(env.config.charge_turns,2)
        obs,info=env.reset(seed=42,options={'scenario':'navigation'})
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(obs['state'].shape,(16,))
        self.assertEqual(info['scenario'],'navigation')
        self.assertEqual(env.robot.position,env.robot.maze.start)

    def test_return_lessons_reachable_charge_then_escape(self):
        lessons=0
        for seed in range(6):
            env=RangeChargeEnv(Config(size=64,charge_turns=2),1)
            obs,info=env.reset(seed=seed)
            if info['scenario']!='return_practice':continue
            lessons+=1
            robot=env.robot
            back,route=energy_route(robot.maze,robot.position,robot.direction,robot.maze.start)
            goal,_=energy_route(robot.maze,robot.position,robot.direction,robot.maze.goal)
            self.assertLess(robot.energy_units,goal)
            self.assertGreaterEqual(robot.energy_units,back)
            for a in route: obs,_,t,tr,info=env.step(a)
            self.assertFalse(t or tr)
            self.assertTrue(info['returned_to_station'])
            obs,_,t,tr,info=env.step(CHARGE)
            self.assertTrue(info['charged_after_return'])
            _,escape=energy_route(robot.maze,robot.position,robot.direction,robot.maze.goal)
            for a in escape: obs,_,t,tr,info=env.step(a)
            self.assertTrue(info['is_success'])
            self.assertTrue(info['success_after_charge'])
            self.assertGreater(info['moves_after_charge'],0)
        self.assertGreater(lessons,0)

    def test_reset_reproducible_and_no_teacher_label_in_input(self):
        env=RangeChargeEnv(Config(size=11,charge_turns=2),1)
        a,ia=env.reset(seed=4);b,ib=env.reset(seed=4)
        self.assertEqual(ia,ib)
        self.assertEqual(set(a),{'local','state'})
        for key in a:np.testing.assert_array_equal(a[key],b[key])

if __name__=='__main__':unittest.main()
