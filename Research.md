# 啓示

1. ヒューリスティック始めての計算後、後続きはもう一度ヒューリスティックスより、変動を計算して元の値に計上可能

2. 結果が少ないと、すべての結果を列挙して保存する。その後ルックアップで進行
3. 记录之前的数据，如果有代价更小的进行更新。
4. メモリーの分配を何度の小規模より一気に大規模
5. 数字をビットとして表示
6. データ構造の工夫、PQよりインデックス、位置情報を表示するリストなど

# 課題１　

+ 1）ヒューリスティック関数を使わないＡ＊ == BFS

１００個のランダムに作成したパズルを解く実行時間：約0.37秒（io出力なし）

パズルごとに、正しい状態（＝ゴール状態）から、一歩ランダムな方向にに空タイルを移動することを三十回繰り返す。つまりランダム移転数：３０。

サンプル：

<img src="../%E5%9B%BE%E7%89%87/%E5%9B%BE%E5%BA%8A/image-20211024094246814.png" alt="image-20211024094246814" style="zoom:33%;" />

+ 2）

ゴールの位置にないタイルの数をヒューリスティックとする

１）と同じ条件として、１００個のランダムに作成したパズルを解く実行時間：0.04秒ほど（io出力なし）

たまに0.1秒が出る時がある。

十回：0.161 0.048 0.041 0.044 0.041 0.039 0.045 0.039 0.038 0.042

+ 3)

マンハッタン距離をヒューリスティックとする

１）と同じ条件として、１００個のランダムに作成したパズルを解く実行時間：0.03秒ほど（io出力なし）

**pythonの場合は1.4 sほど**

十回：0.033 0.031 0.029 0.027 0.031 0.029 0.032 0.045 0.030 0.027

# 課題１A

## C++

```c++
#include<iostream>
#include<math.h>
#include<vector>
#include<memory>
#include<time.h>
#include<queue>
#include<time.h>
#define block_length 4
using std::cout;
using std::vector;
using std::unique_ptr;
typedef vector<vector<int>> Board;
typedef vector<vector<int>> *ptr_to_b;
typedef vector<int> position;
int hamming(Board board);
int manhattan(Board board);
void print_board(Board board);
int single_manhattan(int x,int y,int value);
Board exchange(ptr_to_b p,int i,int j,int ii,int jj);



Board exchange(ptr_to_b p,int i,int j,int ii,int jj)
{
    Board new_board=*p;
    int temp=new_board[i][j];
    new_board[i][j]=new_board[ii][jj];
    new_board[ii][jj]=temp;
    return new_board;
}

void print_board(Board b)
{
    for (int i=0;i<block_length;i++){
        for (int j=0;j<block_length;j++){
            cout<<b[i][j]<<" ";
        }
        cout<<"\n";
    }
    cout<<"\n";
}

class Node{
public:
    int moves;
    int prior;
    int mand;
    position my_blank;
    Board my_board;
    Node *my_parent;
    Node(Board board,position blank){
        my_board=board;
        my_parent=nullptr;
        moves=0;
        mand = manhattan(board);
        prior=moves+mand;
        my_blank=blank;
    }
    Node(Board board,Node *parent,position blank){
        int parent_x,parent_y,old_mand,new_mand;
        my_board=board;
        my_parent=parent;
        my_blank=blank;
        moves=my_parent->moves+1;
        parent_x = parent->my_blank[0];
        parent_y = parent->my_blank[1];
        old_mand =single_manhattan(blank[0],blank[1],parent->my_board[blank[0]][blank[1]]);
        new_mand =single_manhattan(parent_x,parent_y,my_board[parent_x][parent_y]);
        mand = parent->mand-old_mand+new_mand;
        prior=moves+mand;
    }
    ~Node(){
    }
};

Node random_initial();
void print_path(Node n);

struct compare //重写仿函数
{
    bool operator() (Node* a, Node* b)
    {
        return a->prior > b->prior; //minimum heap
    }
};

std::priority_queue<Node*,std::vector<Node*>, compare> Queue;

int hamming(Board board)      //the number of tiles in wrong position
{
    int hamming=0;
    for (int i=0;i<block_length;i++){
        for (int j=0;j<block_length;j++){
            if (board[i][j] != 0){
                hamming+=board[i][j]!=i*block_length+j+1?1:0;
            }
        }
    }
    return hamming;
}

int manhattan(Board board){
    int manhattan = 0;
    for (int i=0;i<block_length;i++){
        for (int j=0;j<block_length;j++){
            int temp=board[i][j];
            if (temp != 0)
            {
                manhattan+=single_manhattan(i,j,temp);
            }
        }
    }
    return manhattan;
}

int single_manhattan(int x,int y,int value){
    int right_x = (value-1)/block_length;
    int right_y = (value-1)%block_length;
    int distance;
    distance=abs(x-right_x)+abs(y-right_y);
    return distance;
}

void next_step(Node *current, int *counter){
    Board board=current->my_board;
    bool has_parent=false;
    Board parent;
    if (current->my_parent){
        parent=current->my_parent->my_board;
        has_parent=true;
    }
    Board new_board;
    for (int i=0;i<block_length;i++){
        for (int j=0;j<block_length;j++){
            if (board[i][j]==0){
                position cur_pos = current->my_blank;
                if (i > 0) { //up
                    new_board = exchange(&board, i,j,i-1,j);
                    if (!has_parent || has_parent && new_board!=parent){
                        position new_pos1 = cur_pos;
                        new_pos1[0]--;
                        Queue.push(new Node(new_board,current,new_pos1));
                        *counter+=1;
//                        cout<<"x:"<<new_pos1[0]<<" y:"<<new_pos1[1]<<"\n";
//                        print_board(new_board);
                    }
                }
                if (i < block_length - 1) { //down
                    new_board = exchange(&board, i, j, i + 1, j);
                    if (!has_parent || has_parent && new_board!=parent){
                        position new_pos2 = cur_pos;
                        new_pos2[0]++;
                        Queue.push(new Node(new_board,current,new_pos2));
                        *counter+=1;
                    }
                }
                if (j > 0) { //left
                    new_board = exchange(&board, i, j, i, j - 1);
                    if (!has_parent || has_parent && new_board!=parent){
                        position new_pos3 = cur_pos;
                        new_pos3[1]--;
                        Queue.push(new Node(new_board,current,new_pos3));
                        *counter+=1;
                    }
                }
                if (j < block_length - 1) { //right
                    new_board = exchange(&board, i, j, i, j + 1);
                    if (!has_parent || has_parent && new_board!=parent){
                        position new_pos4 = cur_pos;
                        new_pos4[1]++;
                        Queue.push(new Node(new_board,current,new_pos4));
                        *counter+=1;
                    }
                }
            }
        }
    }
}

Node random_initial(int random_steps){
    int start[block_length][block_length];
    int init[16]={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0};
    position pos;
    pos.push_back(3);
    pos.push_back(3);
    int i,j,random_num;
    for (i=0;i<block_length;i++){
        for (j=0;j<block_length;j++){
            start[i][j]=init[block_length*i+j];
        }
    }
    Board temp(block_length);
    for (i = 0; i < temp.size(); i++)
        temp[i].resize(block_length);

    for(i = 0; i < temp.size(); i++)
    {
        for (j = 0; j < temp[0].size();j++)
        {
            temp[i][j] = start[i][j];
        }
    }
    Board new_board=temp;
    while (random_steps!=0){
        random_steps--;
        random_num=std::rand() % 4; //0:up,1:down,2:left,3:right
        if (pos[0]>0 && random_num==0){
            new_board = exchange(&new_board, pos[0],pos[1],pos[0]-1,pos[1]);
            pos[0]--;
        }
        else if (pos[0]<block_length-1 && random_num==1){
            new_board = exchange(&new_board, pos[0],pos[1],pos[0]+1,pos[1]);
            pos[0]++;
        }
        else if (pos[1]>0 && random_num==2){
            new_board = exchange(&new_board, pos[0], pos[1], pos[0], pos[1] - 1);
            pos[1]--;
        }
        else if (pos[1]<block_length-1 && random_num==3){
            new_board = exchange(&new_board, pos[0], pos[1], pos[0], pos[1] + 1);
            pos[1]++;
        }
        else{
            continue;
        }

    }
    Node initial(new_board,pos);
    return initial;
}

void print_path(Node n){
    cout<<"print path：\n";
    Node current=n;
    vector<Board> path;
    path.push_back(current.my_board);
    while (current.my_parent!=nullptr){
        current=*(current.my_parent);
        path.push_back(current.my_board);
    }
    for (int i=path.size()-1;i>=0;i--)
    {
        print_board(path[i]);
        cout<<"\n";
    }
}

int main(){
    int count_node = 0;
    int loop_times=100;
    double dur;
    clock_t start,end;
    start=clock();
    Node target=random_initial(0);
    while (loop_times--){
        std::srand(loop_times);
        Node start=random_initial(30);//移动步数
        next_step(&start, &count_node);
    	Node* temp=Queue.top();
        while (temp->my_board != target.my_board){
            Queue.pop();
            next_step(temp, &count_node);
            temp=Queue.top();
        }
//            cout<<"find result!moves:"<<temp->moves<<"\n";
//            print_path(*temp);
		while (!Queue.empty())
	{
		    Queue.pop();
		}
    }
    end=clock();
    dur=(double)(end-start);
    cout<<"Use Time:"<<dur/CLOCKS_PER_SEC<<"\n";
    cout<<"the number of nodes: "<<count_node<<"\n";
    cout<<"generating nodes per sec:"<<count_node/(dur/CLOCKS_PER_SEC);
    return 0;
}
```

## python version

```python
import numpy as np
import random
import time
from collections import defaultdict
from heapq import *
import copy


BLOCK_LENGTH = 5


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
                    right_x = (temp - 1) // BLOCK_LENGTH
                    right_y = (temp - 1) % BLOCK_LENGTH
                    value += abs(i - right_x) + abs(j - right_y)
        return value

    def hamming(self):
        value = 0
        b = self.board.get_board()
        for i in range(BLOCK_LENGTH):
            for j in range(BLOCK_LENGTH):
                temp = b[i][j]
                if temp != 0:
                    value += 1 if temp != i * BLOCK_LENGTH + j + 1 else 0
        return value

    def get_pv(self):
        return self.moves + self.h


class Board:
    def __init__(self):
        self.board = np.append(np.arange(1, BLOCK_LENGTH**2), 0).reshape(BLOCK_LENGTH, BLOCK_LENGTH)
        self.pos_x = BLOCK_LENGTH-1
        self.pos_y = BLOCK_LENGTH-1

    def random_initialize(self, random_step):
        while random_step != 0:
            pos_x = self.pos_x
            pos_y = self.pos_y
            random_step -= 1
            direction = random.randint(1, 4)  # 1-4 represent 4 directions
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
        start = time.perf_counter()
        loop_times = 100
        for i in range(loop_times):
            start_b = Board()
            start_b.random_initialize(30)
            start_s = State(start_b)
            cur_state = start_s
            while np.allclose(cur_state.board.get_board(), target_board) is False:
                self.next_step(cur_state)
                cur_state = self.pq.pop_()
            # self.print_path(cur_state)

        end = time.perf_counter()
        print("{}times span:{:.2f}".format(loop_times, end-start))


sol = PuzzleSolver()
sol.main()
```

# 課題２

+ result: msk_009問題集1,011門を全問解くのに要した時間が4秒

+ Some of the efforts: 

​		constraint porpagation, used min value heuristics when choosing the possible node to expand

## python version

**ソースコード(constraint propagation)**

```python
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
            # delete value range one by one
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
        """
        for ele_list in self.cor_unit[key]:  # row, col, block
            candidates = [s for s in ele_list if num in value_range[s]]  # find candidate place to put num
            if len(candidates) == 0:  # no where to press
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
        for num in value_range[key]:  # min value heuristics
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
```

# 課題3

+ hints: 

  initialize with every row and column exist only one queen, then we can pay attention to conflicts on diagonal and anti-diagonal

+ results:

  + min conflicts(C++): 1000000 queens answer time:0.27s

  + min conflicts(python):1000 queens answer time:0.028212960000000002s

  ​										1000000 queens answer time:10.0174091s

## バックトラック＋制約伝播

### python version

+  結論：n=10レベルの問題なら効率よく解けるが、100レベルになると所要時間が大幅に伸びて、解けなくなった,(n=16の問題を解ける時間は0.4秒ほど、n=8の問題は0.0036秒ほど)。

```python
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


ins = eightqueen_backtrack(16)
ins.execution()
```

## min conflicts

### C++ version

+ source code

```c++
#include <iostream>
#include <ctime>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <random>
using namespace std;

#define MAX 80000000
int board[MAX];
int pos_diag[2 * MAX - 1]; // n queens on main diagonal
int neg_diag[2 * MAX - 1]; // n queens on main anti-diagonal

// better way to generate random number
unsigned long long int RandSeed = (unsigned)time(NULL);
unsigned long long int get_randindex(long long int n) {
	unsigned long long int x;
	double i;
	
	x = 0x7fffffff;
	x += 1;
	RandSeed *= ((unsigned long long int)134775813);
	RandSeed += 1;
	RandSeed = RandSeed % x;
	i = ((double)RandSeed) / (double)0x7fffffff;
	
	return (unsigned long long int)(n * i);
}

int random_start_qs4(int n, int c) {
    int m = n - c;
    for (int i = 0; i < n; ++i) board[i] = i; //one queen every column, there are no coflicts on columns later since we'll exchange column by column
    memset(pos_diag, 0, sizeof(int) * (2 * n - 1));
    memset(neg_diag, 0, sizeof(int) * (2 * n - 1));
    int conflicts = 0;
    int j;
    // place m queens in spare column for guarantee no conlicts
    for (int i = 0, last = n; i < m; ++i, --last) {
        // choose j from [i,n), ensure don't affect the queens which are already placed
        j = i + get_randindex(last);
        while (pos_diag[i - board[j] + n - 1] > 0 || neg_diag[i + board[j]] > 0) j = i + get_randindex(last);
        swap(board[i], board[j]);
        pos_diag[i - board[i] + n - 1]++; //初始化时候的放置不计入在内，此时的放置才视为有效 
        neg_diag[i + board[i]]++;
    }
    // The remaining c queens are placed casually, regardless of whether there is a conflict or not
    for (int i = m, last = c; i < n; ++i, --last) {
        j = i + get_randindex(last);
        swap(board[i], board[j]);
        pos_diag[i - board[i] + n - 1]++;
        neg_diag[i + board[i]]++;
    }
    for (int i = 0; i < 2 * n - 1; ++i) { //overall conficts num on board
        conflicts += pos_diag[i] > 1 ? pos_diag[i] - 1 : 0;
        conflicts += neg_diag[i] > 1 ? neg_diag[i] - 1 : 0;
    }
    cout<<"初始化一次\n";
    return conflicts;
}


// 初始时冲突的皇后数量,这里的数字是指随机放置的皇后个数,不管是否产生冲突
int get_c(int n) {
    if (n <= 10) return n > 8 ? 8 : n;
    else if (n <= 100) return 30;
    else if (n <= 10000) return 50;
    else if (n <= 100000) return 80;
    return 100;
}

// 交换第i和第j个皇后带来的冲突数变化
int swap_gain(int i, int j, int n) {
    int gain = 0;
    // 原来位置对应的对角线上皇后数需要减1
    // 如果原来对应的对角线上有冲突,则gain--
    if (neg_diag[i + board[i]] > 1) gain--;
    if (neg_diag[j + board[j]] > 1) gain--;
    if (pos_diag[i - board[i] + n - 1] > 1) gain--;
    if (pos_diag[j - board[j] + n - 1] > 1) gain--;
    // 同理可知新对应的对角线上有皇后,则gain++
    if (neg_diag[i + board[j]] > 0) gain++;
    if (neg_diag[j + board[i]] > 0) gain++;
    if (pos_diag[i - board[j] + n - 1] > 0) gain++;
    if (pos_diag[j - board[i] + n - 1] > 0) gain++;
    return gain;
}

// 由于交换而更新冲突表和皇后位置
void update_state(int i, int j, int n) {
    neg_diag[i + board[i]]--;
    neg_diag[j + board[j]]--;
    pos_diag[i - board[i] + n - 1]--;
    pos_diag[j - board[j] + n - 1]--;

    neg_diag[i + board[j]]++;
    neg_diag[j + board[i]]++;
    pos_diag[i - board[j] + n - 1]++;
    pos_diag[j - board[i] + n - 1]++;
    
    swap(board[i], board[j]);
}

void local_search(int n, int c) {
    bool restart = true;
    int curr;
    clock_t start = clock();
    int m = n - c;
    while (true) {
        if (restart) curr = random_start_qs4(n, c);
        if (curr <= 0) break;
        restart = true;
        int gain = 0;
        // 随机交换两个皇后,第一个从产生冲突的里面选取,第二个完全随机选取
        for (int i = 0; i < n; ++i) {
            if (pos_diag[i - board[i] + n - 1] > 1 || neg_diag[i + board[i]] > 1) {
                for (int j = 0; j < n; ++j) {
                    if (i != j) {
                        gain = swap_gain(i, j, n);
                        if (gain < 0) {
                            update_state(i, j, n);
                            curr += gain;
                            restart = false;
                            break;
                        }
                    }
                }
                if (restart) break;
            }
        }
    }
    clock_t end = clock();
    cout << "solved in " << (double)(end - start) / CLOCKS_PER_SEC << "s" << endl;
}

int main() {
    srand(unsigned(time(0)));
    int n;
    while (true) {
        cout << "input (pos)num of queens, -1 to quit" << endl;
        cin >> n;
        if (n <= 0 || n > MAX) break;
        for (int run = 0; run < 10; run++) {
            local_search(n, get_c(n));
            // print result when n is small
            if (n <= 10) {
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        if (board[i] == j) cout << "Q";
                        else cout << "+";
                    }
                    cout << endl;
                }
            }
        }
    }
    return 0;
}
```

### python version

+ source code

```python
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
        self.mdiag = np.array([0] * (2 * n - 1))  # rest matrix
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
        # random.seed()
        restart = True
        curr = float("inf")
        while True:
            if restart:
                curr = self.initialize(self.get_c(self.n))
            if curr <= 0:  # no conlifcts,quit
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
                    if restart:  # no more columns to exchange
                        break
        print("final result:")
        if self.n <= 10:
            self.print_board()

def main():
    start = time.perf_counter()
    loops = 1
    for i in range(loops):
        n = 1000000
        NQ = NQueens(n)
        NQ.compute()
    duration = time.perf_counter() - start
    print(f"{n} queens answer time:{duration/loops}s")

main()
```

# 课题4

まず比較結果を示す、どちらもpythonを用いて行うこと

| Queens_num   | 4      | 8      | 16     | 100         |
| ------------ | ------ | ------ | ------ | ----------- |
| 自作ソルバー | 0.0005 | 0.0036 | 0.4276 | -(解けない) |
| SATソルバー  | 0.0001 | 0.0007 | 0.0132 | 0.9561      |

+ 今回使ったのはpython-satというライブラリのGlucose3ソルバーで、読み込みの形式はDIMACSの`p cnf nbvar nbcluase`を省略した以外はだいだい同じ流れです。

  形式について、4 Queensを例としてあげます

  ```python
  # 行制約
  1 2 3 4 0
  -1 -2 0
  -1 -3 0
  -1 -4 0
  -2 -3 0
  -2 -4 0
  -3 -4 0
  5 6 7 8 0
  ....
  9 10 11 12 0
  ....
  13 14 15 16 0
  ...
  # 列制約
  c <Column Contraints>
  1 5 9 13 0
  -1 -5 0
  -1 -9 0
  -1 -13 0
  -5 -9 0
  -5 -13 0
  -9 -13 0
  2 6 10 14 0
  ...
  3 7 11 15 0
  ...
  4 8 12 16 0
  ...
  # 対角線制約
  c <Diagnoal Contraints(skewed to left)>
  -1 -6 0
  -1 -11 0
  -1 -16 0
  -6 -11 0
  -6 -16 0
  -11 -16 0
  -2 -7 0
  -2 -12 0
  -7 -12 0
  -5 -10 0
  -5 -15 0
  -10 -15 0
  -3 -8 0
  -9 -14 0
  c <Diagnoal Contraints(skewed to right)>
  -4 -7 0
  -4 -10 0
  -4 -13 0
  -7 -10 0
  -7 -13 0
  -10 -13 0
  -3 -6 0
  -3 -9 0
  -6 -9 0
  -8 -11 0
  -8 -14 0
  -11 -14 0
  -2 -5 0
  -12 -15 0
  ```

## python version

```python
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
    print(g.get_model()) # 結果を出力


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
```

# 课题5

| Solver\puzzle search time(s)                                 | eight01 | eight02 | eight03 |
| ------------------------------------------------------------ | ------- | ------- | ------- |
| Fast Downward(blind heuristic)                               | 0.63    | 0.64    | 0.63    |
| Fast Downward(iPDB heuristic with default settings)          | 0.06    | 0.06    | 0.08    |
| Fast Downward(landmark-cut heuristic)                        | 0.74    | 0.72    | 0.78    |
| Fast Downward(Lazy greedy best-first search)<br />additive heuristic | 0.04    | 0.06    | 0.04    |
| Fast Downward(Lazy greedy best-first search)<br />FF heuristic | 0.03    | 0.02    | 0.02    |
| 自作ソルバー(python)                                         | 2.90    | 2.40    | 2.07    |
| 自作ソルバー(C++)                                            | 0.12    | 0.19    | 0.11    |

# 课题6

未优化的点： 1. 适应性函数 2. pbest选择 3. r2选择，未使用储备池



https://github.com/P-N-Suganthan/CEC2014

[(28条消息) 优化算法——差分进化算法(DE)_null的专栏-CSDN博客_差分进化算法](https://felix.blog.csdn.net/article/details/41247753?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~Rate-1.pc_relevant_paycolumn_v3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~Rate-1.pc_relevant_paycolumn_v3&utm_relevant_index=2)

[(28条消息) 多目标优化的遗传算法及其改进(浮点数编码)，对多个函数进行测试_天才在于积累-CSDN博客_schaffer函数](https://blog.csdn.net/yanguilaiwuwei/article/details/46699801)

[黑盒优化简介 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/66312442)

[(27条消息) 常见测试函数_千寻的博客-CSDN博客_测试函数](https://blog.csdn.net/jiang425776024/article/details/87358300)

首先c源代码中input_data，M_1_D10：旋转矩阵，1表示第几个函数，10表示维度，10*10的矩阵

shift_data_1: 移动后的全局最优值，代表$o$

shuffle_data_1_D_10: 第1个函数，维度为10，值的含义？

​								有的函数用了有的函数没用？？

|                    | 轮数 | 种群大小 | 准确率(误差0.1以下) | 总时长 | 500轮每轮运行时长/s |
| ------------------ | ---- | -------- | ------------------- | ------ | ------------------- |
| 遗传算法           | 1000 | 200      | 99.4%               |        | 0.35                |
| JADE(算数平均,p=1) | 500  | 200      | 100%                |        | 0.77                |
| SHADE              | 200  | 200      | 100%                |        | 1.02                |

# 论文

+ planning

  https://www.aaai.org/Library/SOCS/socs-library.php  students papers 刚好,其他的也需要看

  http://icaps22.icaps-conference.org/    载完了

  https://www.aaai.org/Library/ICAPS/icaps-library.php

+ 进化计算

  https://cec2021.mini.pw.edu.pl/

  https://gecco-2021.sigevo.org/HomePage

  https://ppsn2022.cs.tu-dortmund.de/

$$
P:(M,S,L,A)
\\M:grid\;map
\\S:start\;point
\\L:Line\;of\;Sight(LOS)\;Function\;like\;4-way\;LOS;8-way\;LOS
\\A:the\;action\;watchman\;can\;choose:1-cell\;movement
$$



$$
Green:current\;position,\;Gray:the\;cells\;have\;been\;seen,\;Red:pivot
\\Yellow:LOS\;of\;pivot,\;White:the\;cells\;have\;been\;ignored\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;
$$

# 研究方向

可以适用于各种LOS函数，无需回到原点，移动之间的cost设计为距离?

(加上时间限制是否可行)

+ Watchman问题先行研究

  两篇论文：https://ojs.aaai.org/index.php/SOCS/article/view/18557/18346

  https://ojs.aaai.org/index.php/ICAPS/article/view/6668/6522

  基准问题测试数据集：https://movingai.com/benchmarks/grids.html

  MST:[scipy.sparse.csgraph.minimum_spanning_tree — SciPy v1.8.0 Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.minimum_spanning_tree.html#:~:text=A minimum spanning tree is,New in version 0.11)

## 先行研究

+ summary：https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.85.9494&rep=rep1&type=pdf



是否存在时间限制的相关论文，如TSP问题，判断基准是什么

TSPTW：https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.421.2268&rep=rep1&type=pdf

TDTSP: https://link.springer.com/chapter/10.1007/0-387-23529-9_12

## Socs调研

[Vol. 12 No. 1 (2021): Fourteenth International Symposium on Combinatorial Search | Proceedings of the International Symposium on Combinatorial Search (aaai.org)](https://ojs.aaai.org/index.php/SOCS/issue/view/445)

1. **基于矛盾的搜索 Conflict based search(在很多论文提到多次)**  MVC(minimal vertex cover)

   问题可以查一下，论文先放着

2. 最优算法的参数自动配置问题 irace

3. 利用ML去解决搜索算法问题，如TSP问题。设计神经网络来解决这类问题

   利用神经网络近似policy function以及heuristic funciton解决15puzzle问题

4. 不仅要最小化代价，同时也要缩短计算时间，所以选择正确的展开节点就很重要，比起通过单纯的期待值去选择展开节点，根据belief distribution去选择更加有效（想法来自RL） metareasoning

5. 同样也是meta reasoning，在蒙特卡洛树搜索等的应用，将强化学习的返回期待值替换成分布

   经典问题：AE2,ACE2，在有限时间内解决问题。用meta reasoning去解决各种问题(multi-agent)，也提到了MVC

6. 和1很像，提到了相同的问题(MAPF)，以及CBS，但对Multi-Agent Path Finding进行了再定义，允许agent移动部分阻碍

7. new lifted heuristic based on landmark

   提出了一种新的启发式函数？

## icaps调研

1. 启发式知识用于加速RL学习，RL的长期奖赏过于稀疏，但启发式知识可以有效解决这些问题。
2. 利用LSTM估计还有多久能够得出答案

## Gecco

1. Genetic algorithm niching by (Quasi-)infinite memory

   利用布隆过滤器作为存储历史，来得到更高的效率



