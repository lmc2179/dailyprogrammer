import unittest
import numpy as np

class IDDQDSolver(object):
    def solve(self, M):
        """
        M is a binary numpy matrix. Returns the number of max hits.
        """
        best_solution = self._get_optimal_column_solution(M)
        for rotation in range(3):
            M = self._rotate_90_right(M)
            best_solution = max(best_solution, self._get_optimal_column_solution(M))
        return best_solution

    def _get_optimal_column_solution(self, M):
        row_count, col_count = M.shape
        best_solution_hits = float('-inf')
        current_hits = [0] * col_count
        for r, row in enumerate(M):
            for c, element in enumerate(row):
                if c + 1 < col_count and row[c+1] == 1:
                    current_hits[c] += 1
                if c > 0 and row[c-1] == 1:
                    current_hits[c] += 1
                if element == 1:
                    current_hits[c] += 1
                    if current_hits[c] > best_solution_hits:
                        best_solution_hits = current_hits[c]
                    current_hits[c] = 0
        return best_solution_hits

    def _rotate_90_right(self, M):
        M_rows, M_cols = M.shape
        M_rotated = np.zeros((M_cols, M_rows))
        for r in range(M_rows):
            for c in range(M_cols):
                M_rotated[c][-r] = M[r][c]
        return M_rotated

class IDDQDSolverTest(unittest.TestCase):
    def test_4_x_1(self):
        M = np.array([[0],
                      [0],
                      [0],
                      [1]])
        for m in [M, M.T]:
            result = IDDQDSolver().solve(m)
            self.assertEqual(result, 1)

    def test_4_x_2(self):
        M = np.array([[1, 0],
                      [0, 0],
                      [0, 1],
                      [1, 1]])
        for m in [M, M.T]:
            result = IDDQDSolver().solve(m)
            self.assertEqual(result, 3)

    def test_sample(self):
        s = """6 10
2 4
4 6
5 5
0 0
0 6"""
        M = parse_description(s)
        for m in [M, M.T]:
            result = IDDQDSolver().solve(m)
            self.assertEqual(result, 4)

    def test_challenge(self):
        s = """20 20
11 16
5 19
12 5
8 9
0 10
17 16
14 9
10 8
19 7
17 11
6 10
0 4
10 9
11 13
19 6
17 10
8 11
6 0
18 17
2 10
12 11
4 2
1 0
2 17
10 5
8 3
13 14
10 14
4 3
5 2"""
        M = parse_description(s)
        result = IDDQDSolver().solve(M)
        self.assertEqual(result, 11)

def parse_description(s):
    lines = s.split('\n')
    row_count, col_count = [int(x) for x in lines[0].split(' ')]
    M = np.zeros((row_count, col_count))
    for line in lines[1:]:
        r,c = [int(x) for x in line.split(' ')]
        M[r][c] = 1
    return M


if __name__ == '__main__':
    unittest.main()