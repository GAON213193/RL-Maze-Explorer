"""Encode only public RobotEnv observations for Gymnasium/SB3."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from .robot import Config, RobotEnv

class MazeGymEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, config=None):
        self.config = config or Config()
        self.robot = RobotEnv(self.config)
        n = self.config.size
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict({
            'memory': spaces.Box(0, 1, (n*n*5,), np.float32),
            'visible': spaces.Box(0, 1, (n*n,), np.float32),
            'state': spaces.Box(0, 1, (11,), np.float32)})

    def _encode(self, obs):
        n = obs['size']
        memory = np.eye(5, dtype=np.float32)[np.asarray(obs['memory']) + 1]
        state = [*(np.asarray(obs['position']) / (n-1)),
                 *(np.asarray(obs['start']) / (n-1)),
                 *np.eye(4, dtype=np.float32)[obs['direction']],
                 obs['battery'] / obs['battery_capacity'],
                 obs['battery_capacity'] / (1 + obs['battery_capacity']),
                 float(obs['collision'])]
        return {'memory': memory.reshape(-1),
                'visible': np.asarray(obs['visible'], dtype=np.float32).reshape(-1),
                'state': np.asarray(state, dtype=np.float32)}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        maze_seed = int(self.np_random.integers(0, 2**31 - 1))
        return self._encode(self.robot.reset(seed=maze_seed)), {}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError('action must be an integer in [0, 3]')
        obs, reward, terminated, truncated, info = self.robot.step(int(action))
        info['is_success'] = info['reason'] == 'success'
        info['turns'] = self.robot.turns
        return self._encode(obs), float(reward), terminated, truncated, info
