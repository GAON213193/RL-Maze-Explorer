"""Dependency-free simulator. observation() is the policy boundary.

Full maze, oracle, trajectory, and debug_stats are evaluator-only data.
Directions: north=0, east=1, south=2, west=3. Energy is in half units internally.
"""
from dataclasses import dataclass
from copy import deepcopy
from heapq import heappush, heappop
import math
from .maze_generator import Maze, generate_maze

DIRECTIONS = ((0,-1), (1,0), (0,1), (-1,0))
FORWARD, LEFT, RIGHT, CHARGE = range(4)
UNKNOWN, OPEN, WALL, GOAL, STATION = -1, 0, 1, 2, 3

@dataclass(frozen=True)
class Config:
    size: int = 64
    algorithm: str = 'dfs'
    view_distance: int = 5
    battery_margin: float = 1.25
    charge_turns: int = 5
    charge_threshold: float = 1.0  # Legacy configs retain their original charging rule.
    max_turns: int = 20000
    collision_cost: float = 1.0
    step_penalty: float = -0.01
    collision_penalty: float = -0.05
    new_cell_reward: float = 0.01
    success_reward: float = 10.0
    depletion_penalty: float = -1.0

    def __post_init__(self):
        for name in ('size', 'view_distance', 'charge_turns', 'max_turns'):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(name + ' must be a positive integer')
        if not math.isfinite(self.charge_threshold) or not 0 <= self.charge_threshold <= 1:
            raise ValueError('charge_threshold must be between 0 and 1')
        if self.size < 5 or self.algorithm not in ('dfs', 'prim'):
            raise ValueError('invalid size or algorithm')
        if not math.isfinite(self.battery_margin) or self.battery_margin < 1:
            raise ValueError('battery_margin must be finite and >= 1')
        if not math.isfinite(self.collision_cost) or self.collision_cost < 0 or (2*self.collision_cost)%1:
            raise ValueError('collision_cost must be a nonnegative multiple of 0.5')
        for name in ('step_penalty','collision_penalty','new_cell_reward','success_reward','depletion_penalty'):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(name + ' must be finite')


def inward_direction(maze):
    x,y = maze.start
    if y == 0: return 2
    if x == maze.size-1: return 3
    if y == maze.size-1: return 0
    if x == 0: return 1
    return 0


def minimum_turn_route(maze, direction):
    """Evaluator oracle: minimize turns, then energy among equal-turn routes."""
    source = (*maze.start, direction)
    heap = [(0, 0, source)]
    best = {source:(0,0)}
    previous = {}
    while heap:
        turns, energy, state = heappop(heap)
        if best[state] != (turns, energy): continue
        x,y,d = state
        if (x,y) == maze.goal:
            actions=[]
            while state != source:
                state,action = previous[state]
                actions.append(action)
            return turns, energy/2, actions[::-1]
        dx,dy = DIRECTIONS[d]
        options = [(LEFT,(x,y,(d-1)%4),1),(RIGHT,(x,y,(d+1)%4),1)]
        if maze.is_open((x+dx,y+dy)):
            options.append((FORWARD,(x+dx,y+dy,d),2))
        for action,nxt,cost in options:
            score=(turns+1,energy+cost)
            if score < best.get(nxt,(math.inf,math.inf)):
                best[nxt]=score;previous[nxt]=(state,action)
                heappush(heap,(*score,nxt))
    raise ValueError('goal is unreachable')


def line_visible(maze, start, target):
    """Supercover ray: walls are visible, cells behind them are not.

    At diagonal corner crossings either touching wall blocks the ray.
    """
    x,y=start;tx,ty=target
    nx,ny=abs(tx-x),abs(ty-y)
    sx=1 if tx>x else -1;sy=1 if ty>y else -1
    ix=iy=0
    while ix<nx or iy<ny:
        decision=(1+2*ix)*ny-(1+2*iy)*nx
        if decision == 0:
            if not maze.is_open((x+sx,y)) or not maze.is_open((x,y+sy)):
                return False
            x+=sx;y+=sy;ix+=1;iy+=1
        elif decision < 0: x+=sx;ix+=1
        else: y+=sy;iy+=1
        if (x,y)==target: return True
        if not maze.is_open((x,y)): return False
    return True


class RobotEnv:
    def __init__(self, config=None):
        self.config=config or Config()
        self.maze=None

    def reset(self, seed=None, maze=None, direction=None):
        self.maze=deepcopy(maze) if maze is not None else generate_maze(
            self.config.size,self.config.algorithm,seed)
        n=self.maze.size
        if any(len(row)!=n or any(v not in (0,1) for v in row) for row in self.maze.grid):
            raise ValueError('maze must be square and binary')
        self.position=self.maze.start
        self.direction=inward_direction(self.maze) if direction is None else direction
        if self.direction not in range(4) or not self.maze.is_open(self.position) or not self.maze.is_open(self.maze.goal):
            raise ValueError('invalid pose or endpoints')
        if self.maze.start == self.maze.goal: raise ValueError('start and goal must differ')
        turns,energy,_=minimum_turn_route(self.maze,self.direction)
        self.oracle_turns=turns;self.oracle_energy=energy
        self.capacity_units=math.ceil(energy*2*self.config.battery_margin)
        self.energy_units=self.capacity_units
        self.turns=self.collisions=self.charges=0
        self.reason=None;self.terminated=self.truncated=False
        self.memory=[[UNKNOWN]*n for _ in range(n)]
        # Station is known initially, even before observing any nearby cells.
        x,y=self.position;self.memory[y][x]=STATION
        self.last_collision=False
        self.trajectory=[self.position]
        self._observe()
        return self.observation()

    @property
    def battery(self): return self.energy_units/2
    @property
    def capacity(self): return self.capacity_units/2

    def _observe(self):
        n=self.maze.size;px,py=self.position;dx,dy=DIRECTIONS[self.direction]
        r=self.config.view_distance
        self.visible=set();new=0
        for y in range(max(0,py-r),min(n,py+r+1)):
            for x in range(max(0,px-r),min(n,px+r+1)):
                ax,ay=x-px,y-py
                forward=ax*dx+ay*dy;side=abs(ax*dy-ay*dx)
                if (x,y)!=self.position and (forward<=0 or side>forward or ax*ax+ay*ay>r*r):continue
                if not line_visible(self.maze,self.position,(x,y)):continue
                self.visible.add((x,y))
                if self.memory[y][x]==UNKNOWN:new+=1
                self.memory[y][x]=(STATION if (x,y)==self.maze.start else
                    GOAL if (x,y)==self.maze.goal else self.maze.grid[y][x])
        return new

    def observation(self):
        if self.maze is None:raise RuntimeError('call reset first')
        n=self.maze.size
        visible=[[int((x,y) in self.visible) for x in range(n)] for y in range(n)]
        # No hidden goal, ground-truth grid, oracle path or oracle cost.
        return dict(size=n, start=list(self.maze.start), position=list(self.position),
                    direction=self.direction, memory=deepcopy(self.memory), visible=visible,
                    battery=self.battery, battery_capacity=self.capacity,
                    collision=self.last_collision)

    def step(self,action):
        if self.maze is None:raise RuntimeError('call reset first')
        if self.terminated or self.truncated:raise RuntimeError('episode ended; reset required')
        if isinstance(action,bool) or not isinstance(action,int) or action not in range(4):
            raise ValueError('action must be 0,1,2,3')
        self.last_collision=False;duration=1;reward=self.config.step_penalty
        if action==CHARGE and self.position==self.maze.start and self.energy_units<self.capacity_units and self.energy_units<=self.capacity_units*self.config.charge_threshold:
            # Atomic full charge; cannot finish if the episode time runs out first.
            duration=min(self.config.charge_turns,self.config.max_turns-self.turns)
            reward=self.config.step_penalty*duration
            if duration==self.config.charge_turns:
                self.energy_units=self.capacity_units;self.charges+=1
        elif action==CHARGE:
            # Outside station or above charge threshold; rejected attempt still costs one turn.
            reward+=self.config.collision_penalty
        else:
            dx,dy=DIRECTIONS[self.direction];x,y=self.position
            target=(x+dx,y+dy)
            hit=action==FORWARD and not self.maze.is_open(target)
            cost=int(2*self.config.collision_cost) if hit else (2 if action==FORWARD else 1)
            if self.energy_units<cost:
                self.terminated=True;self.reason='insufficient_energy'
            else:
                self.energy_units-=cost
                if action==LEFT:self.direction=(self.direction-1)%4
                elif action==RIGHT:self.direction=(self.direction+1)%4
                elif hit:self.last_collision=True;self.collisions+=1;reward+=self.config.collision_penalty
                else:self.position=target
        self.turns+=duration;self.trajectory.append(self.position)
        reward+=self._observe()*self.config.new_cell_reward
        if self.position==self.maze.goal:
            self.terminated=True;self.reason='success';reward+=self.config.success_reward
        elif self.energy_units==0 and self.position!=self.maze.start:
            self.terminated=True;self.reason='battery_depleted'
        if self.reason in ('insufficient_energy','battery_depleted'):reward+=self.config.depletion_penalty
        if not self.terminated and self.turns>=self.config.max_turns:
            self.truncated=True;self.reason='turn_limit'
        return self.observation(),reward,self.terminated,self.truncated,{'reason':self.reason,'elapsed_turns':duration}

    def debug_stats(self):
        return dict(turns=self.turns,battery=self.battery,capacity=self.capacity,
                    collisions=self.collisions,charges=self.charges,reason=self.reason,
                    oracle_min_turns=self.oracle_turns,oracle_energy=self.oracle_energy)
