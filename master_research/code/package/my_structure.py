from collections import defaultdict
from heapq import *


class PriorityQueue:
    def __init__(self):
        self.pq = defaultdict(list)
        self.min_pos = []

    def push_(self, state):
        value = state.get_value()
        self.pq[value].append(state)
        heappush(self.min_pos, value)

    def pop_(self):
        assert not self.is_empty(), "队列已空"
        return self.pq[heappop(self.min_pos)].pop()

    def is_empty(self):
        return True if len(self.min_pos) == 0 else False


class Myarray:
    def __init__(self, shape):
        self.shape = shape
        self.n_row, self.n_col = shape
        self.list = [[] for i in range(self.n_row*self.n_col)]

    def __setitem__(self, key, value):
        row, col = key
        idx = row * self.n_col + col
        self.list[idx] = value

    def __getitem__(self, item):
        row, col = item
        idx = row*self.n_col+col
        return self.list[idx]

    def __str__(self):
        s = ""
        for i in range(self.n_row):
            for j in range(self.n_col):
                s += str(self.__getitem__((i, j))) + ' '
            s += '\n'
        return s