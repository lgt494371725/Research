import numpy as np
import time
from pysat.solvers import Glucose3


class SATSolver:
    def __init__(self, n):
        self.n = n
        self.str_array = []

    def make_n_queens_array(self):
        return np.arange(1, self.n * self.n + 1).reshape(self.n, self.n)

    def repeat(self, alist):
        length = len(alist)
        for x in range(length - 1):
            for y in range(x + 1, length):
                self.str_array.append(str(-alist[x]) + " " + str(-alist[y]) + " 0")

    def row_constraints(self, board):
        for alist in board:
            temp_str = ' '.join(map(str, alist)) + " 0"
            self.str_array.append(temp_str)
            self.repeat(alist)

    def col_constraints(self, board):
        self.str_array.append("c <Column Contraints>")
        self.row_constraints(board.T)

    def diag_contraints(self, board):
        self.str_array.append("c <Diagnoal Contraints(skewed to left)>")
        temp_array = [np.diag(board, 0)]
        for i in range(1, self.n - 1):
            temp_array.extend([np.diag(board, i), np.diag(board, -i)])
        for alist in temp_array:
            self.repeat(alist)
        self.str_array.append("c <Diagnoal Contraints(skewed to right)>")
        temp_array = [np.diag(np.fliplr(board), 0)]
        for i in range(1, self.n - 1):
            temp_array.extend([np.diag(np.fliplr(board), i), np.diag(np.fliplr(board), -i)])
        for alist in temp_array:
            self.repeat(alist)

    def run(self):
        board = self.make_n_queens_array()
        self.row_constraints(board)
        self.col_constraints(board)
        self.diag_contraints(board)


def execute(array):
    g = Glucose3()
    for sen in array:
        g.add_clause(sen)
    print(g.solve())
    print(g.get_model())


def read_file(path):
    sentences = []
    with open(path, mode='r', encoding='utf8') as f:
        for line in f.readlines():
            if line[0] != 'c':
                sentences.append(list(map(int, line.strip('\n').split()))[:-1])
    return sentences


def make_file(array, path):
    with open(path, 'w+') as f:
        for lines in array:
            f.write(lines)
            f.write('\n')


def main():
    n = 100
    path = str(n)+'_queen.txt'
    solver = SATSolver(n)
    solver.run()
    make_file(solver.str_array, path)
    array = read_file(path)
    start = time.perf_counter()
    execute(array)
    print(f"process time {time.perf_counter()-start}")



main()