"""Near junction detail (0..3 forward cells), far axial range only."""
from .robot import RobotEnv, DIRECTIONS, UNKNOWN, OPEN, WALL, GOAL, STATION


class RangeRobotEnv(RobotEnv):
    def _observe(self):
        n = self.maze.size
        px, py = self.position
        dx, dy = DIRECTIONS[self.direction]
        rx, ry = -dy, dx
        self.visible = set()
        self.forward_range = 0
        new = 0

        def reveal(x, y, detail):
            nonlocal new
            if not (0 <= x < n and 0 <= y < n):
                return
            self.visible.add((x, y))
            if self.memory[y][x] == UNKNOWN:
                new += 1
            if detail:
                self.memory[y][x] = (STATION if (x, y) == self.maze.start else
                                     GOAL if (x, y) == self.maze.goal else self.maze.grid[y][x])
            elif self.memory[y][x] == UNKNOWN:
                # Far range reveals traversability, not new semantic landmarks.
                self.memory[y][x] = self.maze.grid[y][x]

        reveal(px, py, True)
        # Immediate left/right openings are visible, but nothing beyond that cell.
        for side in (-1, 1):
            reveal(px + side*rx, py + side*ry, True)
        for distance in range(1, n):
            x, y = px + distance*dx, py + distance*dy
            if not (0 <= x < n and 0 <= y < n):
                break
            reveal(x, y, distance <= 3)
            if not self.maze.is_open((x, y)):
                break
            self.forward_range = distance
            if distance <= 3:
                for side in (-1, 1):
                    reveal(x + side*rx, y + side*ry, True)
        return new

    def observation(self):
        obs = super().observation()
        obs['forward_range'] = self.forward_range
        return obs
