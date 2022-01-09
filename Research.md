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

+  結論：n=10レベルの問題なら効率よく解けるが、100レベルになると所要時間が大幅に伸びて、解けなくなった,(n=16の問題を解ける時間は0.4秒ほど、n=8の問題は0.008秒ほど)。

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

https://www.cs.drexel.edu/~jjohnson/2017-18/fall/CS270/Lectures/8/sat.pdf

+ 4queens sample

  ```python
  ;; Definitions of the 16 boolean variables
  (declare-const x0y0 Bool)
  (declare-const x0y1 Bool)
  (declare-const x0y2 Bool)
  (declare-const x0y3 Bool)
  (declare-const x1y0 Bool)
  (declare-const x1y1 Bool)
  (declare-const x1y2 Bool)
  (declare-const x1y3 Bool)
  (declare-const x2y0 Bool)
  (declare-const x2y1 Bool)
  (declare-const x2y2 Bool)
  (declare-const x2y3 Bool)
  (declare-const x3y0 Bool)
  (declare-const x3y1 Bool)
  (declare-const x3y2 Bool)
  (declare-const x3y3 Bool)
  
  ;;"at least one queen by line" clauses
  (assert (or x0y0  x0y1  x0y2 x0y3))
  (assert (or x1y0  x1y1  x1y2 x1y3))
  (assert (or x2y0  x2y1  x2y2 x2y3))
  (assert (or x3y0  x3y1  x3y2 x3y3))
  
  ;;"only one queen by line" clauses
  
  (assert (not (or(and x0y1 x0y0)(and x0y2 x0y0)(and x0y2 x0y1)(and x0y3 x0y0)(and x0y3 x0y1)(and x0y3 x0y2))))
  (assert (not (or(and x1y1 x1y0)(and x1y2 x1y0)(and x1y2 x1y1)(and x1y3 x1y0)(and x1y3 x1y1)(and x1y3 x1y2))))
  (assert (not (or(and x2y1 x2y0)(and x2y2 x2y0)(and x2y2 x2y1)(and x2y3 x2y0)(and x2y3 x2y1)(and x2y3 x2y2))))
  (assert (not (or(and x3y1 x3y0)(and x3y2 x3y0)(and x3y2 x3y1)(and x3y3 x3y0)(and x3y3 x3y1)(and x3y3 x3y2))))
  
  ;;"only one queen by column" clauses
  (assert (not (or(and x1y0 x0y0)(and x2y0 x0y0)(and x2y0 x1y0)(and x3y0 x0y0)(and x3y0 x1y0)(and x3y0 x2y0))))
  (assert (not (or(and x1y1 x0y1)(and x2y1 x0y1)(and x2y1 x1y1)(and x3y1 x0y1)(and x3y1 x1y1)(and x3y1 x2y1))))
  (assert (not (or(and x1y2 x0y2)(and x2y2 x0y2)(and x2y2 x1y2)(and x3y2 x0y2)(and x3y2 x1y2)(and x3y2 x2y2))))
  (assert (not (or(and x1y3 x0y3)(and x2y3 x0y3)(and x2y3 x1y3)(and x3y3 x0y3)(and x3y3 x1y3)(and x3y3 x2y3))))
  
  ;;"only one queen by diagonal" clauses
  (assert (not (or (and x0y0 x1y1) (and x0y0 x2y2) (and x0y0 x3y3) (and x1y1 x2y2) (and x1y1 x3y3) (and x2y2 x3y3))))
  (assert (not (or (and x0y1 x1y2) (and x0y1 x2y3) (and x1y2 x2y3))))
  (assert (not (or (and x0y2 x1y3))))
  (assert (not (or (and x1y0 x2y1) (and x1y0 x3y2) (and x2y1 x3y2))))
  (assert (not (or (and x2y0 x3y1))))
  (assert (not (or (and x1y1 x0y0) (and x2y2 x0y0) (and x3y3 x0y0) (and x2y2 x1y1) (and x3y3 x1y1) (and x3y3 x2y2))))
  (assert (not (or (and x1y2 x0y1) (and x2y3 x0y1) (and x2y3 x1y2))))
  (assert (not (or (and x1y3 x0y2))))
  (assert (not (or (and x2y1 x1y0) (and x3y2 x1y0) (and x3y2 x2y1))))
  (assert (not (or (and x3y1 x2y0))))
  
  ;; Check if the generate model is satisfiable and output a model.
  (check-sat)
  (get-model)
  ```

+ n queens

```python
import os
import sys

if len(sys.argv) < 2:
    sys.exit('Usage: %s <problem size>' % sys.argv[0])

def nl(f):
    f.write('\n')

# Output file
filename = '%s_queens_SAT.smt2' % sys.argv[1]
f = open(filename, 'w')

N = int(sys.argv[1])
print "Opening %s to write the SMT-LIB v2 encoding of the %i-queens problem" % (filename, N)

f.write(';; Generate the definitions of the variables\n')
for i in range(0, N):
    for j in range(0,N):
        f.write('(declare-const x%iy%i Bool)\n' % (i, j))

f.write(';;Generate the "one queen by line" clauses\n\n')
for i in range(0,N):
    f.write('(assert (or')
    for j in range(0, N-1):
        f.write(' x%iy%i ' % (i,j))
    f.write('x%iy%i' %(i, N-1))
    f.write('))')
    f.write('\n')


f.write('\n;;Generate the "only one queen by line" clauses\n\n')
for i in range(0,N):
    f.write('(assert (not (or')
    for j in range(1, N):
        for k in range(0,j):
            f.write('(and x%iy%i x%iy%i)' %(i,j,i,k))
    f.write(')))')
    nl(f)
nl(f)

f.write(';;Generate the "only one queen by column" clauses\n')
for i in range(0,N):
    f.write('(assert (not (or')
    for j in range(1, N):
        for k in range(0,j):
            f.write('(and x%iy%i x%iy%i)' %(j,i,k,i))

    f.write(')))')
    nl(f)
nl(f)

f.write(';;Generate the "only one queen by diagonal" clauses\n')
for i in range(0,N-1):
    f.write('(assert (not (or')
    for j in range(0, N-i):
        for k in range(1,N-j-i):
            f.write(' (and x%iy%i x%iy%i)' %(j,i+j,j+k,i+j+k))
    f.write(')))')
    nl(f)

for i in range(1,N-1):
    f.write('(assert (not (or')
    for j in range(0, N-i):
        for k in range(1,N-j-i):
            f.write(' (and x%iy%i x%iy%i)' %(i+j,j,i+j+k,j+k))
    f.write(')))')
    nl(f)

for i in range(0,N-1):
    f.write('(assert (not (or')
    for j in range(0, N-i):
        for k in range(1,N-j-i):
            f.write(' (and x%iy%i x%iy%i)' %(j+k,i+j+k,j,i+j))
    f.write(')))')
    nl(f)

for i in range(1,N-1):
    f.write('(assert (not (or')
    for j in range(0, N-i):
        for k in range(1,N-j-i):
            f.write(' (and x%iy%i x%iy%i)' %(i+j+k,j+k,i+j,j))
    f.write(')))')
    nl(f)

nl(f)


f.write(";; Check if the generate model is satisfiable and output a model.\n")
f.write("(check-sat)\n")
f.write("(get-model)\n")
f.close()

# solution_filename = 's%i_queens.txt' %  N
# os.system('z3 %s > %s' % (filename, solution_filename))
# solution_file = open(solution_filename, 'r')
```

