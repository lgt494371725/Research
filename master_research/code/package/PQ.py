from collections import defaultdict
from heapq import *


class PriorityQueue:
    def __init__(self):
        self.pq = defaultdict(list)
        self.min_pos = []

    def push_(self, state):
        self.pq[state.A_star_v].append(state)
        heappush(self.min_pos, state.A_star_v)

    def pop_(self):
        assert not self.is_empty(), "队列已空"
        return self.pq[heappop(self.min_pos)].pop()

    def is_empty(self):
        return True if len(self.min_pos) == 0 else False