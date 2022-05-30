import numpy as np
import random
import time
from collections import defaultdict
from heapq import *
import copy


BLOCK_LENGTH = 3


class State:
    def __init__(self, board, parent=None):
        self.board = board
        self.parent = parent
        self.moves = 0 if not parent else parent.moves+1
        self.h = self.manhattan()

    def manhattan(self):  # faster
        value = 0
        b = self.board.get_board()
        for i in range(BLOCK_LENGTH):
            for j in range(BLOCK_LENGTH):
                temp = b[i][j]
                if temp != 0:
                    right_x = temp % BLOCK_LENGTH
                    right_y = temp // BLOCK_LENGTH
                    value += abs(i - right_x) + abs(j - right_y)
        return value

    def get_pv(self):
        return self.moves + self.h


class Board:
    def __init__(self):
        """
        [0,3,6]
        [1,4,7]
        [2,5,8]
        """
        self.board = np.arange(3**2).reshape(3, 3).T
        self.pos_x = 0
        self.pos_y = 0

    def random_initialize(self, random_step):
        """
        random_step: 棋盘随机初始化经历的步数
        """
        while random_step != 0:
            pos_x = self.pos_x
            pos_y = self.pos_y
            random_step -= 1
            direction = random.randint(1, 4)  # 1-4代表4个方向
            # left
            if pos_x > 0 and direction == 1:
                self.move(pos_x, pos_y, pos_x-1, pos_y)
            # right
            elif pos_x < BLOCK_LENGTH - 1 and direction == 2:
                self.move(pos_x, pos_y, pos_x + 1, pos_y)
            elif pos_y > 0 and direction == 3:
                self.move(pos_x, pos_y, pos_x, pos_y-1)
            elif pos_y < BLOCK_LENGTH - 1 and direction == 4:
                self.move(pos_x, pos_y, pos_x, pos_y+1)
            else:
                continue

    def move(self, i, j, ii, jj):
        self.board[i][j], self.board[ii][jj] = \
            self.board[ii][jj], self.board[i][j]
        self.pos_x = ii
        self.pos_y = jj

    def print_board(self):
        print("blank_position:", f"{self.pos_x},{self.pos_y}")
        for i in range(BLOCK_LENGTH):
            print(self.board[i])
        print("\n")

    def get_board(self):
        return self.board

    def get_blank(self):
        return self.pos_x, self.pos_y

    def init_board(self):
        # puzzle 3
        # self.pos_x = 2
        # self.pos_y = 2
        # self.board = np.array([[8,7,4], [5,2,1], [6,3,0]])
        # puzzle 2
        # self.pos_x = 1
        # self.pos_y = 0
        # self.board = np.array([[8,5,2], [0,4,3], [6,7,1]])
        # puzzle 1
        self.pos_x = 0
        self.pos_y = 1
        self.board = np.array([[8,0,2], [7,4,5], [6,1,3]])


class PriorityQueue:
    def __init__(self):
        self.pq = defaultdict(list)
        self.min_pos = []

    def push_(self, state):
        self.pq[state.get_pv()].append(state)
        heappush(self.min_pos, state.get_pv())

    def pop_(self):
        return self.pq[heappop(self.min_pos)].pop()

    def is_empty(self):
        return True if len(self.min_pos) == 0 else False


class PuzzleSolver:
    def __init__(self):
        self.pq = PriorityQueue()

    def print_path(self, state):
        paths = []
        while state.parent:
            paths.append(state.board)
            state = state.parent
        print(f"total steps:{len(paths)}")
        for board in paths[::-1]:
            board.print_board()

    def next_step(self, state):
        parent_sta = state.parent
        x, y = state.board.get_blank()
        if x > 0:
            cur_board = copy.deepcopy(state.board)
            cur_board.move(x, y, x-1, y)
            if not parent_sta or np.allclose(cur_board.get_board(), parent_sta.board.get_board()) is False:
                self.pq.push_(State(cur_board, parent=state))
        if x < BLOCK_LENGTH-1:
            cur_board = copy.deepcopy(state.board)
            cur_board.move(x, y, x+1, y)
            if not parent_sta or np.allclose(cur_board.get_board(), parent_sta.board.get_board()) is False:
                self.pq.push_(State(cur_board, parent=state))
        if y > 0:
            cur_board = copy.deepcopy(state.board)
            cur_board.move(x, y, x, y-1)
            if not parent_sta or np.allclose(cur_board.get_board(), parent_sta.board.get_board()) is False:
                self.pq.push_(State(cur_board, parent=state))
        if y < BLOCK_LENGTH-1:
            cur_board = copy.deepcopy(state.board)
            cur_board.move(x, y, x, y+1)
            if not parent_sta or np.allclose(cur_board.get_board(), parent_sta.board.get_board()) is False:
                self.pq.push_(State(cur_board, parent=state))

    def main(self):
        target = Board()
        target.random_initialize(0)
        target_board = target.get_board()
        start_b = Board()
        start_b.init_board()
        # start_b.random_initialize(100)
        start_s = State(start_b)
        cur_state = start_s
        start = time.perf_counter()
        while np.allclose(cur_state.board.get_board(), target_board) is False:
            self.next_step(cur_state)
            cur_state = self.pq.pop_()
        self.print_path(cur_state)
        end = time.perf_counter()
        print("times span:{:.3f}".format(end-start))


sol = PuzzleSolver()
sol.main()