import unittest
from environment.maze_generator import generate_maze,shortest_path,try_add_wall

class MazeTests(unittest.TestCase):
    def test_boundaries_and_connectivity(self):
        for size in (7,8,16,32,64):
            for algo in ('dfs','prim'):
                for seed in range(4):
                    with self.subTest(size=size,algo=algo,seed=seed):
                        m=generate_maze(size,algo,seed)
                        border={(x,y) for y in range(size) for x in range(size)
                                if (x in (0,size-1) or y in (0,size-1)) and m.grid[y][x]==0}
                        self.assertEqual(border,{m.start,m.goal})
                        sides=lambda p: (p[1]==0,p[0]==size-1,p[1]==size-1,p[0]==0)
                        self.assertEqual(sum(sides(m.start)),1)
                        self.assertEqual(sum(sides(m.goal)),1)
                        self.assertNotEqual(sides(m.start),sides(m.goal))
                        self.assertGreaterEqual(len(shortest_path(m))-1,size)
    def test_reproducible(self):
        self.assertEqual(generate_maze(seed=17),generate_maze(seed=17))
    def test_invalid_constraint(self):
        with self.assertRaises(ValueError):generate_maze(8,min_path_length=1000)
    def test_protected_cells(self):
        m=generate_maze(16,seed=5)
        for p in (m.start,m.goal):self.assertFalse(try_add_wall(m,p))
    def test_reject_disconnect(self):
        m=generate_maze(16,seed=2);p=shortest_path(m);cut=p[len(p)//2]
        self.assertFalse(try_add_wall(m,cut));self.assertTrue(m.is_open(cut))
