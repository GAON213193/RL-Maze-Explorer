"""Version 2: size-independent local memory and exploration rewards.

Only RobotEnv's public observation is encoded. The exit is never revealed early.
Visited cells and turn streak are explicit episode memory, retained on charging.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from .robot import Config, RobotEnv, FORWARD, LEFT, RIGHT, CHARGE

PROFILE = 'exploration-v2'

class ExplorationEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, config=None):
        self.config = config or Config(size=7, max_turns=196)
        self.robot = RobotEnv(self.config)
        self.action_space = spaces.Discrete(4)
        # Egocentric 9x9 crop: 5 memory categories, visibility, visited cells.
        self.observation_space = spaces.Dict({
            'local': spaces.Box(0, 1, (9*9*7,), np.float32),
            'state': spaces.Box(-1, 1, (15,), np.float32)})
        self.visited = set()
        self.no_move = self.turn_streak = self.forward_moves = 0

    def _encode(self, obs):
        n = obs['size']; px,py = obs['position']
        forward = ((0,-1),(1,0),(0,1),(-1,0))[obs['direction']]
        right = (-forward[1],forward[0])
        local = np.zeros((9,9,7), dtype=np.float32)
        for row in range(9):
            for col in range(9):
                ahead,side = 4-row,col-4
                x = px + ahead*forward[0] + side*right[0]
                y = py + ahead*forward[1] + side*right[1]
                if 0 <= x < n and 0 <= y < n:
                    local[row,col,obs['memory'][y][x]+1] = 1
                    local[row,col,5] = obs['visible'][y][x]
                    local[row,col,6] = float((x,y) in self.visited)
                else:
                    local[row,col,2] = 1  # Known map boundary, not hidden terrain.
        sx,sy = obs['start']; dx,dy = sx-px,sy-py
        state = [px/(n-1), py/(n-1), sx/(n-1), sy/(n-1),
                 *np.eye(4)[obs['direction']], obs['battery']/obs['battery_capacity'],
                 obs['battery_capacity']/(1+obs['battery_capacity']), float(obs['collision']),
                 (dx*forward[0]+dy*forward[1])/(2*(n-1)),
                 (dx*right[0]+dy*right[1])/(2*(n-1)),
                 min(self.turn_streak,24)/24, min(self.no_move,24)/24]
        return {'local':local.reshape(-1), 'state':np.asarray(state,dtype=np.float32)}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.robot.reset(seed=int(self.np_random.integers(0,2**31-1)))
        self.visited = {tuple(obs['position'])}
        self.no_move = self.turn_streak = self.forward_moves = 0
        return self._encode(obs), {}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError('action must be an integer in [0,3]')
        action = int(action)
        before = self.robot.position
        obs, _, terminated, truncated, info = self.robot.step(action)
        moved = self.robot.position != before
        self.forward_moves += int(moved)
        first_visit = moved and self.robot.position not in self.visited
        self.visited.add(self.robot.position)
        self.no_move = 0 if moved else self.no_move+1
        self.turn_streak = self.turn_streak+1 if action in (LEFT,RIGHT) else 0
        # Rotations needed for navigation (including a U-turn) have no extra cost.
        reward = -0.002*info['elapsed_turns']
        if first_visit: reward += 0.05
        if obs['collision']: reward -= 0.02
        if self.turn_streak > 2: reward -= 0.02
        if action == CHARGE and (before != self.robot.maze.start or info['elapsed_turns']==1):
            reward -= 0.02
        if info['reason'] == 'success': reward += 10
        elif info['reason'] in ('battery_depleted','insufficient_energy'): reward -= 2
        elif truncated: reward -= 2
        elif self.no_move >= 24:
            truncated = True
            self.robot.truncated = True
            self.robot.reason = info['reason'] = 'stalled'
            reward -= 2
        info.update(is_success=info['reason']=='success', turns=self.robot.turns,
                    unique_cells=len(self.visited), forward_moves=self.forward_moves)
        return self._encode(obs), float(reward), terminated, truncated, info
