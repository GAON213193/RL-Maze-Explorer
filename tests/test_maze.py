import unittest

from environment.maze_generator import generate_maze, shortest_path, try_add_wall


class MazeTests(unittest.TestCase):
    def test_connected_and_bordered(self):
        for size in (8, 16, 32, 64):
            for algorithm in ("dfs", "prim"):
                maze = generate_maze(size, algorithm, seed=42)
                self.assertEqual(len(maze.grid), size)
                self.assertTrue(all(len(row) == size for row in maze.grid))
                self.assertIsNotNone(shortest_path(maze))
                self.assertTrue(all(cell == 1 for cell in maze.grid[0]))
                self.assertTrue(all(cell == 1 for cell in maze.grid[-1]))
                self.assertTrue(all(row[0] == 1 and row[-1] == 1 for row in maze.grid))

    def test_reproducible(self):
        self.assertEqual(generate_maze(32, seed=17).grid, generate_maze(32, seed=17).grid)

    def test_editor_does_not_block_goal_or_agent(self):
        maze = generate_maze(16, seed=5)
        self.assertFalse(try_add_wall(maze, maze.goal))
        self.assertFalse(try_add_wall(maze, maze.start, agent_position=maze.start))
        self.assertIsNotNone(shortest_path(maze))

    def test_reject_disconnect(self):
        maze = generate_maze(16, seed=2)
        path = shortest_path(maze)
        # In a perfect maze, any interior cell of the unique solution path is a cut.
        cut = path[len(path) // 2]
        self.assertFalse(try_add_wall(maze, cut))
        self.assertTrue(maze.is_open(cut))


if __name__ == "__main__":
    unittest.main()
