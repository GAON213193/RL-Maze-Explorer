import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from environment.gym_env import MazeGymEnv
from environment.robot import Config

class TrainingTests(unittest.TestCase):
    def test_seed_and_partial_observation(self):
        env = MazeGymEnv(Config(size=16))
        obs, _ = env.reset(seed=42)
        again, _ = env.reset(seed=42)
        for key in obs:
            np.testing.assert_array_equal(obs[key], again[key])
        memory = obs['memory'].reshape(16, 16, 5)
        self.assertGreater(memory[:, :, 0].sum(), 0)
        np.testing.assert_array_equal(memory.argmax(axis=2)-1, env.robot.observation()['memory'])
        self.assertEqual(set(obs), {'memory', 'visible', 'state'})
        self.assertTrue(env.observation_space.contains(obs))
        env.close()

    def test_train_save_reload_and_resume_64(self):
        torch.set_num_threads(1)
        for algorithm in ('dfs', 'prim'):
            with self.subTest(algorithm=algorithm):
                env = MazeGymEnv(Config(size=64, algorithm=algorithm, max_turns=64))
                check_env(env, warn=True)
                model = PPO('MultiInputPolicy', env, n_steps=64, batch_size=32,
                            n_epochs=1, seed=42, device='cpu',
                            policy_kwargs=dict(net_arch=dict(pi=[32], vf=[32])))
                before = {k: v.clone() for k, v in model.policy.state_dict().items()}
                model.learn(128)
                self.assertTrue(any(not torch.equal(before[k], v) for k,v in model.policy.state_dict().items()))
                self.assertTrue(all(torch.isfinite(v).all() for v in model.policy.state_dict().values()))
                obs, _ = env.reset(seed=7)
                expected, _ = model.predict(obs, deterministic=True)
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / 'model.zip'
                    model.save(path)
                    loaded = PPO.load(path, env=env, device='cpu')
                    actual, _ = loaded.predict(obs, deterministic=True)
                    np.testing.assert_array_equal(expected, actual)
                    for k,v in model.policy.state_dict().items():
                        self.assertTrue(torch.equal(v, loaded.policy.state_dict()[k]))
                    loaded.learn(64, reset_num_timesteps=False)
                    self.assertEqual(loaded.num_timesteps, 192)
                env.close()

if __name__ == '__main__':
    unittest.main()
