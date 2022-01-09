import numpy as np
import time


class eightqueen_backtrack:
    def __init__(self, n):
        self.n = n
        self.count = 0
        self.cb = np.zeros((n, n))
        self.lastqueen = [-1 for i in range(n)]  # record the last column of n queens
        # lastqueen[0]=1 represents the 0-th queen in the column 1
        self.last_pos = -1  # last position of queen
        self.last_row = [] # the order to place queen

    def search_col(self, x, y):  # search_col(1,2) search from 1-th row and start from col 2
        for col in range(y, self.n):
            if self.cb[x][col] == 0:
                return col
        return -1  # fail to find position

    def set_queen(self, x, y):
        self.last_pos = -1  # reset position
        self.lastqueen[x] = y  # record the set pos
        self.last_row.append(x)
        self.cb[:, y] += 1  # update col
        self.cb[x, :] += 1  # update row
        # update diag
        pos_x, pos_y = x, y
        while pos_x >= 0 and pos_y >= 0:  # upper left
            self.cb[pos_x][pos_y] += 1
            pos_x, pos_y = pos_x - 1, pos_y - 1
        pos_x, pos_y = x, y
        while pos_x < self.n and pos_y < self.n:  # lower right
            self.cb[pos_x][pos_y] += 1
            pos_x, pos_y = pos_x + 1, pos_y + 1
        pos_x, pos_y = x, y
        while pos_x >= 0 and pos_y < self.n:  # lower left
            self.cb[pos_x][pos_y] += 1
            pos_x, pos_y = pos_x - 1, pos_y + 1
        pos_x, pos_y = x, y
        while pos_x < self.n and pos_y >= 0:  # upper right
            self.cb[pos_x][pos_y] += 1
            pos_x, pos_y = pos_x + 1, pos_y - 1
        self.cb[x][y] = -1
        # print(self.cb)
        for i in range(self.n):
            if self.lastqueen[i] == -1 and np.count_nonzero(self.cb[i, :]) == self.n:
                return False
            if self.lastqueen[i] == -1 and self.n - np.count_nonzero(self.cb[i, :]) == 1:
                zero_pos = self.cb[i, :].tolist().index(0)
                return self.set_queen(i, zero_pos)
        return True


    def uptake(self, x, y):
        self.last_pos = self.lastqueen[x]  # mark the pos of queen，prevent duplication
        self.lastqueen[x] = -1  # reset
        self.cb[:, y] -= 1
        self.cb[x, :] -= 1
        pos_x, pos_y = x, y
        while pos_x >= 0 and pos_y >= 0:
            self.cb[pos_x][pos_y] -= 1
            pos_x, pos_y = pos_x - 1, pos_y - 1
        pos_x, pos_y = x, y
        while pos_x < self.n and pos_y < self.n:
            self.cb[pos_x][pos_y] -= 1
            pos_x, pos_y = pos_x + 1, pos_y + 1
        pos_x, pos_y = x, y
        while pos_x >= 0 and pos_y < self.n:
            self.cb[pos_x][pos_y] -= 1
            pos_x, pos_y = pos_x - 1, pos_y + 1
        pos_x, pos_y = x, y
        while pos_x < self.n and pos_y >= 0:
            self.cb[pos_x][pos_y] -= 1
            pos_x, pos_y = pos_x + 1, pos_y - 1
        self.cb[x][y] = 0

    def backtrack(self):
        row = 0
        status = True
        while True:
            if all(self.lastqueen[i] != -1 for i in range(self.n)):  # finish conditon
                temp = self.cb.copy()
                temp[temp != -1] = 0  # For convenient look
                print(temp)
                break
            while self.lastqueen[row] != -1:
                row += 1
            col = self.search_col(row, self.last_pos + 1)
            if col == -1 or not status:
                status = True
                last_x = self.last_row.pop(-1)
                self.uptake(last_x, self.lastqueen[last_x])
                row = last_x-1
            else:
                status = self.set_queen(row, col)
                row = -1
            row += 1

    def execution(self):
        start = time.perf_counter()
        self.backtrack()
        print('{} s'.format(time.perf_counter() - start))


ins = eightqueen_backtrack(100)
ins.execution()