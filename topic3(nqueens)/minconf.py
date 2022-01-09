import random
import time
import numpy as np


class NQueens:
    def __init__(self, n):
        self.board = []
        self.n = n
        self.queens = [-1] * n
        self.mdiag = np.array([0] * (2 * n - 1))
        self.adiag = np.array([0] * (2 * n - 1))

    def initialize(self, c):
        n = self.n
        m = n - c
        self.queens = [i for i in range(n)]
        self.mdiag = np.array([0] * (2 * n - 1))  # 需要对这些矩阵都进行重置
        self.adiag = np.array([0] * (2 * n - 1))
        for i in range(m):
            j = random.randint(i, n - 1)
            while self.mdiag[self.getMaindiag(i, self.queens[j])] > 0 or \
                    self.adiag[self.getAntidiag(i, self.queens[j])] > 0:
                j = random.randint(i, n - 1)
            self.queens[i], self.queens[j] = self.queens[j], self.queens[i]
            self.mdiag[self.getMaindiag(i, self.queens[i])] += 1
            self.adiag[self.getAntidiag(i, self.queens[i])] += 1
            i += 1
        for i in range(m,n):
            j = random.randint(i, n - 1)
            self.queens[i], self.queens[j] = self.queens[j], self.queens[i]
            self.mdiag[self.getMaindiag(i, self.queens[i])] += 1
            self.adiag[self.getAntidiag(i, self.queens[i])] += 1
        conflicts = np.sum(self.mdiag[self.mdiag > 1] - 1) + np.sum(self.adiag[self.adiag > 1] - 1)
        print("initial complete!")
        if self.n <= 10:
            self.print_board()
        return conflicts

    def getAntidiag(self, row, col):
        return row + col

    def getMaindiag(self, row, col):
        return row - col + self.n - 1

    def print_board(self):
        for i in range(self.n):
            value = self.queens[i]
            string = " ." * value + " Q" + " ." * (self.n - value - 1)
            print(string)

    def get_c(self, n):
        if n <= 10:
            return 8 if n > 8 else n
        elif n <= 100:
            return 30
        elif n <= 10000:
            return 50
        elif n <= 100000:
            return 80
        else:
            return 100

    def swap_gain(self, i, j):
        gain = 0
        if self.adiag[self.getAntidiag(i, self.queens[i])] > 1: gain -= 1
        if self.adiag[self.getAntidiag(j, self.queens[j])] > 1: gain -= 1
        if self.mdiag[self.getMaindiag(i, self.queens[i])] > 1: gain -= 1
        if self.mdiag[self.getMaindiag(j, self.queens[j])] > 1: gain -= 1
        if self.adiag[self.getAntidiag(i, self.queens[j])] > 0: gain += 1
        if self.adiag[self.getAntidiag(j, self.queens[i])] > 0: gain += 1
        if self.mdiag[self.getMaindiag(i, self.queens[j])] > 0: gain += 1
        if self.mdiag[self.getMaindiag(j, self.queens[i])] > 0: gain += 1
        return gain

    def update_state(self, i, j):
        self.adiag[self.getAntidiag(i, self.queens[i])] -= 1
        self.adiag[self.getAntidiag(j, self.queens[j])] -= 1
        self.mdiag[self.getMaindiag(i, self.queens[i])] -= 1
        self.mdiag[self.getMaindiag(j, self.queens[j])] -= 1

        self.adiag[self.getAntidiag(i, self.queens[j])] += 1
        self.adiag[self.getAntidiag(j, self.queens[i])] += 1
        self.mdiag[self.getMaindiag(i, self.queens[j])] += 1
        self.mdiag[self.getMaindiag(j, self.queens[i])] += 1
        self.queens[i], self.queens[j] = self.queens[j], self.queens[i]

    def compute(self):
        random.seed()
        restart = True
        curr = float("inf")
        while True:
            if restart:
                curr = self.initialize(self.get_c(self.n))
            if curr <= 0:  # 无矛盾，直接退出
                break
            restart = True
            for i in range(self.n):
                if self.mdiag[self.getMaindiag(i, self.queens[i])] > 1 or \
                        self.adiag[self.getAntidiag(i, self.queens[i])] > 1:
                    for j in range(self.n):
                        if i != j:
                            gain = self.swap_gain(i, j)
                            if gain < 0:
                                self.update_state(i, j)
                                curr += gain
                                restart = False
                                break
                    if restart:  # 说明没有找到可交换的列了
                        break
        print("final result:")
        if self.n <= 10:
            self.print_board()


def main():
    start = time.perf_counter()
    loops = 1
    for i in range(loops):
        n = 16
        NQ = NQueens(n)
        NQ.compute()
    duration = time.perf_counter() - start
    print(f"{n} queens answer time:{duration/loops}s")

main()

# def set_queen(self, x, y):
#     """
#     对角线:
#     row=0,col=1 第一条  n-1,row=0,col=2  第二条  n-2 -> No.=col-row
#     row=1,col=0 row=2,col=1   负一条
#     反对角线：
#     row=0 col=6, row=1 第一条
#     row=0,col=5 第二条
#     row=1,col=7,row=2,col=6 负一条  -> No.=n-(row+col)-1
#     """
#     self.board[:, y] += 1
#     diag_num = y-x
#     diag_length = self.n-abs(diag_num)
#     self.board += np.diag([1]*diag_length, k=diag_num)
#     anti_diag_num = self.n-(x+y)-1
#     anti_diag_length = self.n-abs(anti_diag_num)
#     self.board += np.fliplr(np.diag([1]*anti_diag_length,k=anti_diag_num))
#     self.board[x, y] -= 2  # 有2次重复计算
