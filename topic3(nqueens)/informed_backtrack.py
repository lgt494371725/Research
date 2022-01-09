import random
import time
import numpy as np

class NQueens:
    def __init__(self, n):
        self.queens = [-1] * n
        self.n = n
        self.mdiag = [0] * (2 * n - 1)
        self.adiag = [0] * (2 * n - 1)
        self.cols = [0] * n

    def initialize(self):
        n = self.n
        for row in range(n):
            col = random.randint(0, n-1)
            self.queens[row] = col
            self.set_queens(row, -1, col)
        print("initialize complete:")
        self.print_board()

    def getAntidiag(self, row, col):
        return row + col

    def getMaindiag(self, row, col):
        return row - col + self.n - 1

    def set_queens(self, row, pre_col, col):
        self.cols[col] += 1
        self.mdiag[self.getMaindiag(row, col)] += 1
        self.adiag[self.getAntidiag(row, col)] += 1
        if pre_col == -1:
            return
        self.cols[pre_col] -= 1
        self.mdiag[self.getMaindiag(row, pre_col)] -= 1
        self.adiag[self.getAntidiag(row, pre_col)] -= 1

    def print_board(self):
        for i in range(self.n):
            value = self.queens[i]
            string = " ." * value + " Q" + " ." * (self.n - value - 1)
            print(string)

    def possible_pos(self, row):
        candidates = [(self.cols[j] + self.mdiag[self.getMaindiag(row, j)]+
                      self.adiag[self.getAntidiag(row, j)]-3) for j in range(self.n)]
        return np.argsort(candidates)

    def backtrack(self, row):
        if row == n:
            print("find result:")
            self.print_board()
        temp_row = self.vars_left.pop()
        self.vars_done.append(temp_row)
        candidates = self.possible_pos(temp_row)
        pre_col = self.queens[temp_row]
        for v in candidates:
            if v!= pre_col:
                pass
