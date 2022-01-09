import time
import os


class SudokuSolver:
    def __init__(self, puzzle):
        self.puzzle = puzzle  # 094000130000...81 numbers
        self.cols = '123456789'
        self.rows = 'ABCDEFGHI' # A1:sudoku[0][0] A2:sudoku[0][1] B1:sudoku[1][0]
        self.puzzle_no = self.make_dic_key(self.rows, self.cols)  # 81个 No:['A1', 'A2',... 'A9', ... , 'I8', 'I9']
        self.nine_units = self.nine_list(self.cols, self.rows)
        # including 9 rows 9 cols and 9 blocks [['A1','B1'...,'I1'],['A2',...'I2'],...，[A1,..A9],...]
        self.cor_unit = dict((s, [u for u in self.nine_units if s in u]) for s in self.puzzle_no)
        """
        'A1': [['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G1', 'H1', 'I1'],
               ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9'],
               ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']],'B1':xxxx...."""
        # 9+9+9 27 elements correlated including self.
        self.cor_no = dict((s, set(sum(self.cor_unit[s], [])) - {s}) for s in self.puzzle_no)
        # 8+8+4，the elements may affect the value range of this element except self
        # sum([[1,2],[3,4,5]],[])->[1,2,3,4,5]

    def make_dic_key(self, A, B):  # generalize ['A1', 'A2', 'A3',... 'I1', 'I2', 'I3',] dict key lists
        return [a + b for a in A for b in B]

    def nine_list(self, cols, rows):
        list_cols = [self.make_dic_key(rows, c) for c in cols]  # [['A1','B1'...,'I1'],['A2',...'I2'],...]
        list_rows = [self.make_dic_key(r, cols) for r in rows]  # [['A1',...'A9'],['B1',...'B9'],..]
        list_nine_blocks = [self.make_dic_key(r, c) for r in ('ABC', 'DEF', 'GHI') for c in ('123', '456', '789')]
        return list_cols + list_rows + list_nine_blocks

    def initialize(self):
        value_range = dict((s, self.cols) for s in self.puzzle_no)  # {'A1': '123456789', 'A2': '123456789',....}
        puzzle_dict = dict(zip(self.puzzle_no, self.puzzle))  # {'A1': '4', 'A2': '.', ... , 'I8': '.', 'I9': '.'}
        for key, value in puzzle_dict.items():
            # print(key, value)
            if value in self.cols and not self.replace_ele(value_range, key, value):
                return False
        return value_range

    def replace_ele(self, value_range, key, value):  # remove all values in value_range[key] except value
        other_values = value_range[key].replace(value, '')  # all values in value_range[key] except value
        if all(self.delete_num(value_range, key, num) for num in other_values):
            # 一个一个删，删到只剩一个值做特殊处理,一个一个放的理由参考delete_num注释部分
            return value_range
        else:
            return False

    def delete_num(self, value_range, key, num):  # remove num from value_range[key] and check if length is 1
        if num not in value_range[key]:
            return value_range
        value_range[key] = value_range[key].replace(num, '')
        if len(value_range[key]) == 0:  # means illegal puzzle
            return False
        if len(value_range[key]) == 1:  # if only 1 value possible, process correlated position
            only_value = value_range[key]
            # remove only value from cor 20 elements
            if not all(self.delete_num(value_range, s, only_value) for s in self.cor_no[key]):
                return False
        """
        every row or col or block will exist unique value from 1-9
        every time we do a remove operation, means other place may have a chance to place this value
        exp:if I remove 0 from puzzle[0][0], 
                means rows[0] col[0] and block[0] may have a chance to place 0 except puzzle[0][0]
        
        method below also can solve problem but with a low speed to 4s ->28s
        candidates = [key for key in self.cor_no if num in value_range[key]]
        if len(candidates) == 0:
             return False
        elif len(candidates) == 1:
             only_key = candidates[0]
             if not self.replace_ele(value_range, only_key, num):
                return False
        return value_range
        变慢的理由：将cor_unit换成cor_no后变慢，cor_unit是按照row,col,block来进行判断了，即3次判断
        而cor_no是一整个列表，合起来判断一起实际上是不完全的
        """
        for ele_list in self.cor_unit[key]:  # row, col, block
            candidates = [s for s in ele_list if num in value_range[s]]  # find candidate place to put num
            if len(candidates) == 0:  # nowhere to place
                return False
            elif len(candidates) == 1:
                only_key = candidates[0]
                if not self.replace_ele(value_range, only_key, num):
                    return False
        return value_range

    def search_data(self, value_range):  # recursion instead of backtrack
        if value_range is False:
            return False
        if all(len(value_range[s]) == 1 for s in self.puzzle_no):
            return value_range
        _, key = min((len(value_range[key]), key) for key in self.puzzle_no if len(value_range[key]) > 1)
        result_list = []
        for num in value_range[key]:  # 用可能取值最小的优先遍历
            result_list.append(self.search_data(self.replace_ele(value_range.copy(), key, num)))
        return self.find_result(result_list)

    def find_result(self, result_list):
        for result in result_list:
            if result:
                return result
        return False

    def show_data(self, data):
        if not data:
            print('illegal puzzle')
            return
        temp = []
        for key, value in data.items():
            temp.append(value)
        j = 0
        for i in range(0, 9):
            print(temp[j:j + 9])
            j = j + 9
        print("\n")
        return

    def execute(self):
        result = self.search_data(self.initialize())
        return result

def main():
    # os.chdir(r"D:\OneDrive\OneDrive - The University of Tokyo\research\topic2")
    os.chdir(r"D:\OneDrive - The University of Tokyo\research\topic2")
    with open("msk_009.txt", encoding='utf8', mode='r') as f:
        puzzles = f.readlines()
    nums = len(puzzles)
    sum_time = 0
    for i in range(nums-1):
        print("loop:", i)
        puzzle = puzzles[i].rstrip('\n').replace('.', '0')
        solver = SudokuSolver(puzzle)
        start = time.perf_counter()
        result = solver.execute()
        end = time.perf_counter()
        solver.show_data(result)
        sum_time += end-start
    print("1011 puzzles:%f s" % sum_time)


if __name__ == '__main__':
    main()