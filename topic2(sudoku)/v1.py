import time
import numpy as np
import os


class Sudoku_Solver:
    def __init__(self,sudoku):
        self.sudoku = sudoku
        self.row_record = {i: [0] * 10 for i in range(9)}  # row_record[0][1]==1 means 0_th row has own the number 1
        self.col_record = {i: [0] * 10 for i in range(9)}
        self.block_record = {i: [0] * 10 for i in range(9)}  # there is 9 block with each size is 3*3.
        self.last_pos = []
        self.last_value = []

    def is_legal(self, value, x, y):
        if self.row_record[x][value] == 1:
            return False
        if self.col_record[y][value] == 1:
            return False
        n_th = x // 3 * 3 + y // 3
        if self.block_record[n_th][value] == 1:
            return False
        return True

    def init_pos(self):
        for i in range(9):
            for j in range(9):
                value = self.sudoku[i][j]
                if value != 0:
                    self.row_record[i][value] = 1
                    self.col_record[j][value] = 1
                    n_th = i//3*3+j//3
                    self.block_record[n_th][value] = 1

    def dfs_backtrack(self):
        x, y = 0, 0
        tmp = 1  # 填充的数字
        self.init_pos()
        while True:
            if (x, y) == (8, 9):
                print(self.sudoku)
                return
            elif y < 9 and self.sudoku[x][y] == 0:  # 0说明需要填充
                while tmp < 10 and not self.is_legal(tmp, x, y):
                    tmp += 1
                if tmp == 10:  # no legal pos found,backtrack
                    # print(self.last_pos)
                    x, y = self.last_pos[-1]
                    self.sudoku[x][y] = 0
                    tmp = self.last_value[-1] + 1  # +1之后就不会重复原来的值
                    # print("需要回溯：(x,y):{},tmp:{}",(x,y),tmp)
                    self.last_pos.pop(-1)
                    self.last_value.pop(-1)
                    self.row_record[x][tmp-1] = 0
                    self.col_record[y][tmp-1] = 0
                    n_th = x // 3 * 3 + y // 3
                    self.block_record[n_th][tmp-1] = 0
                    continue
                self.last_pos.append((x, y))
                self.last_value.append(tmp)
                self.sudoku[x][y] = tmp
                self.row_record[x][tmp] = 1
                self.col_record[y][tmp] = 1
                n_th = x // 3 * 3 + y // 3
                self.block_record[n_th][tmp] = 1
                y += 1
                tmp = 1
                # print("row_record",self.row_record)
                # print("col_record",self.col_record)
                # print("block",self.block_record)
                # print(self.sudoku)
            elif y == 9:
                x += 1
                y = 0
            else:  # 说明棋盘上已有数字，pass
                y += 1


def main():
    os.chdir(r"D:\OneDrive\OneDrive - The University of Tokyo\research")
    with open("msk_009.txt", encoding='utf8', mode='r') as f:
        puzzles = f.readlines()
    nums = len(puzzles)
    sum_time = 0
    for i in range(nums):
        puzzle = puzzles[i].rstrip('\n').replace('.', '0')
        puzzle = [int(i) for i in list(puzzle)]
        puzzle = np.array(puzzle).reshape(9, 9)
        solver = Sudoku_Solver(puzzle)
        start = time.perf_counter()
        solver.dfs_backtrack()
        end = time.perf_counter()
        sum_time += end-start
        print("%s completes"%i)
    print("1012puzzles:%.2f" % sum_time)
main()
