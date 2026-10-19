"""Deterministic, connected maze generation with a BFS oracle.

Grid encoding: 0 = traversable, 1 = wall.
The full grid and shortest path are environment/evaluation-only information;
they must never be exposed to a partially observable agent.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import random

Point = tuple[int, int]
Grid = list[list[int]]


@dataclass
class Maze:
    grid: Grid
    start: Point
    goal: Point

    @property
    def size(self) -> int:
        return len(self.grid)

    def is_open(self, point: Point) -> bool:
        x, y = point
        return 0 <= x < self.size and 0 <= y < self.size and self.grid[y][x] == 0


def neighbors(point: Point) -> list[Point]:
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def shortest_path(maze: Maze, start: Point | None = None, goal: Point | None = None) -> list[Point] | None:
    """BFS shortest path. For editor validation and evaluation, NOT agent input."""
    source = maze.start if start is None else start
    target = maze.goal if goal is None else goal
    if not maze.is_open(source) or not maze.is_open(target):
        return None
    previous: dict[Point, Point | None] = {source: None}
    queue = deque([source])
    while queue:
        current = queue.popleft()
        if current == target:
            result = []
            while current is not None:
                result.append(current)
                current = previous[current]
            return result[::-1]
        for nxt in neighbors(current):
            if maze.is_open(nxt) and nxt not in previous:
                previous[nxt] = current
                queue.append(nxt)
    return None


def _odd_cells(size: int) -> list[Point]:
    return [(x, y) for y in range(1, size - 1, 2) for x in range(1, size - 1, 2)]


def generate_maze(size: int = 32, algorithm: str = "dfs", seed: int | None = None) -> Maze:
    """Generate a connected perfect maze using randomized DFS or randomized Prim.

    Even-sized grids retain a solid outer border. Cell corridors occupy odd
    coordinates; the unused final row/column remains wall where necessary.
    """
    if size < 5:
        raise ValueError("size must be at least 5")
    if algorithm not in ("dfs", "prim"):
        raise ValueError("algorithm must be 'dfs' or 'prim'")
    rng = random.Random(seed)
    grid = [[1] * size for _ in range(size)]
    cells = set(_odd_cells(size))
    origin = (1, 1)
    visited = {origin}
    grid[1][1] = 0

    def candidates(p: Point) -> list[Point]:
        x, y = p
        return [q for q in ((x + 2, y), (x - 2, y), (x, y + 2), (x, y - 2)) if q in cells and q not in visited]

    def carve(a: Point, b: Point) -> None:
        ax, ay = a
        bx, by = b
        grid[by][bx] = 0
        grid[(ay + by) // 2][(ax + bx) // 2] = 0

    if algorithm == "dfs":
        stack = [origin]
        while stack:
            options = candidates(stack[-1])
            if not options:
                stack.pop()
                continue
            nxt = rng.choice(options)
            carve(stack[-1], nxt)
            visited.add(nxt)
            stack.append(nxt)
    else:
        frontier = [(origin, nxt) for nxt in candidates(origin)]
        while frontier:
            index = rng.randrange(len(frontier))
            parent, nxt = frontier.pop(index)
            if nxt in visited:
                continue
            carve(parent, nxt)
            visited.add(nxt)
            frontier.extend((nxt, q) for q in candidates(nxt))

    # Choose the farthest reachable odd cell as goal to avoid trivial mazes.
    queue = deque([origin])
    distance = {origin: 0}
    while queue:
        current = queue.popleft()
        for nxt in neighbors(current):
            x, y = nxt
            if 0 <= x < size and 0 <= y < size and grid[y][x] == 0 and nxt not in distance:
                distance[nxt] = distance[current] + 1
                queue.append(nxt)
    goal = max((p for p in cells if p in distance), key=lambda p: distance[p])
    maze = Maze(grid, origin, goal)
    assert shortest_path(maze) is not None
    return maze


def try_add_wall(maze: Maze, point: Point, agent_position: Point | None = None) -> bool:
    """Place a wall only if a path from the CURRENT position to goal remains.

    Never pass the oracle result to the agent. On failure, revert the edit.
    """
    if point in (maze.start, maze.goal, agent_position):
        return False
    if not maze.is_open(point):
        return False
    x, y = point
    maze.grid[y][x] = 1
    source = agent_position if agent_position is not None else maze.start
    if shortest_path(maze, source) is None:
        maze.grid[y][x] = 0
        return False
    return True


def render_ascii(maze: Maze, show_path: bool = False) -> str:
    path = set(shortest_path(maze) or []) if show_path else set()
    rows = []
    for y, row in enumerate(maze.grid):
        chars = []
        for x, wall in enumerate(row):
            p = (x, y)
            chars.append("S" if p == maze.start else "G" if p == maze.goal else "#" if wall else "." if p in path else " ")
        rows.append("".join(chars))
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and validate a random maze")
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--algorithm", choices=("dfs", "prim"), default="dfs")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    maze = generate_maze(args.size, args.algorithm, args.seed)
    path = shortest_path(maze)
    print(f"size={maze.size} algorithm={args.algorithm} start={maze.start} goal={maze.goal} shortest_path_steps={len(path) - 1}")
    if not args.quiet:
        print(render_ascii(maze, show_path=True))


if __name__ == "__main__":
    main()
