import unittest
import numpy as np
from environment.gym_v2 import ExplorationEnv
from environment.robot import Config, RIGHT, LEFT, FORWARD
from training.profiles import make_env

class V2Tests(unittest.TestCase):
    def test_shapes_are_size_independent_and_valid(self):
        for size in (7,11,17,33,64):
            env=ExplorationEnv(Config(size=size))
            obs,_=env.reset(seed=3)
            self.assertTrue(env.observation_space.contains(obs))
            self.assertEqual(obs['local'].shape,(567,))
            self.assertEqual(obs['state'].shape,(15,))

    def test_spin_penalty_and_stall(self):
        env=ExplorationEnv(Config(size=64,battery_margin=10))
        env.reset(seed=7)
        _,r1,_,_,_=env.step(RIGHT)
        _,r2,_,_,_=env.step(RIGHT)
        _,r3,_,_,_=env.step(RIGHT)
        self.assertEqual(r1,r2)
        self.assertLess(r3,r2)
        for _ in range(21): obs,r,t,tr,info=env.step(RIGHT)
        self.assertEqual(info['reason'],'stalled')
        self.assertTrue(t or tr)
        self.assertEqual(info['unique_cells'],1)
        self.assertEqual(info['forward_moves'],0)

    def test_first_visit_reward_not_repeatable(self):
        env=ExplorationEnv(Config(size=7,battery_margin=10))
        env.reset(seed=7)
        _,first,_,_,_=env.step(FORWARD)
        env.step(RIGHT);env.step(RIGHT)
        _,repeat,_,_,_=env.step(FORWARD)
        self.assertGreater(first,repeat)
        self.assertEqual(env.forward_moves,2)
        self.assertEqual(len(env.visited),2)

    def test_hidden_grid_not_used_by_encoder(self):
        env=ExplorationEnv(Config(size=7))
        env.reset(seed=3)
        obs=env.robot.observation();before=env._encode(obs)
        env.robot.maze.goal=(-99,-99)
        env.robot.maze.grid=[[0]*7 for _ in range(7)]
        after=env._encode(obs)
        for key in before: np.testing.assert_array_equal(before[key],after[key])

    def test_legacy_viewer_factory(self):
        self.assertEqual(make_env(Config(size=7),'legacy-v1').observation_space['memory'].shape,(245,))

if __name__=='__main__':unittest.main()
