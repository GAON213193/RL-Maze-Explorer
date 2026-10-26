"""Range vision, two-turn recharge, mixed navigation/return practice.

Lesson generation may use evaluator paths to construct a solvable situation.
These paths and the lesson label are NEVER part of policy observations.
"""
import heapq
import math
import numpy as np
from gymnasium import spaces
from .gym_v2 import ExplorationEnv
from .robot_v3 import RangeRobotEnv
from .robot import Config, DIRECTIONS, FORWARD, LEFT, RIGHT, CHARGE

PROFILE = 'range-charge-v3'


def energy_route(maze, position, direction, target):
    """Evaluator-only shortest energy path in half units, with rotations."""
    start = (*position, direction)
    queue = [(0, start)]
    costs, previous = {start: 0}, {}
    while queue:
        cost, state = heapq.heappop(queue)
        if cost != costs[state]: continue
        x, y, d = state
        if (x, y) == target:
            actions = []
            while state != start:
                state, action = previous[state]
                actions.append(action)
            return cost, actions[::-1]
        dx, dy = DIRECTIONS[d]
        options = [(LEFT, (x,y,(d-1)%4), 1), (RIGHT, (x,y,(d+1)%4), 1)]
        if maze.is_open((x+dx,y+dy)):
            options.append((FORWARD, (x+dx,y+dy,d), 2))
        for action, nxt, step in options:
            if cost+step < costs.get(nxt, math.inf):
                costs[nxt] = cost+step
                previous[nxt] = (state, action)
                heapq.heappush(queue, (cost+step, nxt))
    raise ValueError('Target is unreachable')


class RangeChargeEnv(ExplorationEnv):
    def __init__(self, config=None, lesson_probability=0.3):
        config = config or Config(size=64, charge_turns=2, charge_threshold=0.5)
        if config.charge_turns != 2:
            raise ValueError('v3 requires charge_turns=2')
        if not 0 <= lesson_probability <= 1:
            raise ValueError('lesson_probability must be between 0 and 1')
        super().__init__(config)
        self.robot = RangeRobotEnv(config)
        self.lesson_probability = lesson_probability
        self.observation_space.spaces['state'] = spaces.Box(-1, 1, (16,), np.float32)
        self.scenario = 'navigation'
        self.returned = False
        self.charged_after_return = False
        self.left_station = False
        self.moves_after_charge = 0

    def _encode(self, obs):
        encoded = super()._encode(obs)
        encoded['state'] = np.append(encoded['state'], np.float32(obs['forward_range']/(obs['size']-1))).astype(np.float32)
        return encoded

    def reset(self, *, seed=None, options=None):
        # Parent seeds Gymnasium and clears episode memory.
        super().reset(seed=seed)
        choice = (options or {}).get('scenario')
        if choice not in (None, 'navigation', 'return_practice'):
            raise ValueError('Unknown scenario')
        practice = choice == 'return_practice' or (choice is None and self.np_random.random() < self.lesson_probability)
        self.scenario = 'navigation'
        if practice and self._prepare_return():
            self.scenario = 'return_practice'
        self.visited = set(self.robot.trajectory)
        self.no_move = self.turn_streak = self.forward_moves = 0
        self.left_station = self.robot.position != self.robot.maze.start
        self.returned = False
        self.charged_after_return = False
        self.moves_after_charge = 0
        return self._encode(self.robot.observation()), {'scenario': self.scenario}

    def _prepare_return(self):
        """Walk 2..5 cells away, then lower charge enough to require a return.

        No agent learning actions occur here. Memory contains only the observed
        outbound walk. If no suitable feasible state is found, reset to normal.
        """
        robot = self.robot
        original_maze = robot.maze
        target_moves = int(self.np_random.integers(2, 6))
        walked = 0
        previous = None
        for _ in range(target_moves):
            x,y = robot.position
            options = [(d,(x+dx,y+dy)) for d,(dx,dy) in enumerate(DIRECTIONS)
                       if robot.maze.is_open((x+dx,y+dy)) and (x+dx,y+dy) != robot.maze.goal]
            fresh = [item for item in options if item[1] not in robot.trajectory]
            if fresh: options = fresh
            elif previous is not None:
                options = [item for item in options if item[1] != previous] or options
            if not options: break
            direction, target = options[int(self.np_random.integers(len(options)))]
            turns = (direction-robot.direction)%4
            rotations = [LEFT] if turns == 3 else [RIGHT]*turns
            for action in rotations + [FORWARD]:
                if robot.terminated or robot.truncated: break
                robot.step(action)
            if robot.terminated or robot.truncated: break
            previous = (x,y)
            walked += 1
        if walked >= 2 and not (robot.terminated or robot.truncated):
            back_cost, back_actions = energy_route(robot.maze, robot.position, robot.direction, robot.maze.start)
            goal_cost, _ = energy_route(robot.maze, robot.position, robot.direction, robot.maze.goal)
            energy = back_cost + 2  # one unit spare after a correct return
            end_direction = robot.direction
            for action in back_actions:
                if action == LEFT: end_direction = (end_direction-1)%4
                elif action == RIGHT: end_direction = (end_direction+1)%4
            after_charge, _ = energy_route(robot.maze, robot.maze.start, end_direction, robot.maze.goal)
            if energy < goal_cost and energy < robot.capacity_units and after_charge <= robot.capacity_units:
                robot.energy_units = energy
                # Warm-up walk is context, not scored episode time/actions.
                robot.turns = robot.collisions = robot.charges = 0
                robot.last_collision = False
                robot._observe()
                return True
        robot.reset(maze=original_maze)
        return False

    def step(self, action):
        before = self.robot.position
        charged_before = self.robot.charges
        obs,reward,terminated,truncated,info = super().step(action)
        at_station = self.robot.position == self.robot.maze.start
        if self.robot.position != before:
            if self.robot.charges > 0: self.moves_after_charge += 1
            if at_station and self.left_station: self.returned = True
            if not at_station: self.left_station = True
        if self.robot.charges > charged_before and self.returned:
            self.charged_after_return = True
        # No direct bonus for returning or charging: success must follow.
        info.update(scenario=self.scenario, charges=self.robot.charges,
                    returned_to_station=self.returned,
                    charged_after_return=self.charged_after_return,
                    success_after_charge=info['is_success'] and self.robot.charges > 0,
                    moves_after_charge=self.moves_after_charge,
                    charge_completed=self.robot.charges > charged_before)
        return obs,reward,terminated,truncated,info
