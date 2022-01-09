啓示：

1. ヒューリスティック始めての計算後、後続きはもう一度ヒューリスティックスより、変動を計算して元の値に計上可能

2. 結果が少ないと、すべての結果を列挙して保存する。その後ルックアップで進行
3. 记录之前的数据，如果有代价更小的进行更新。
4. メモリーの分配を何度の小規模より一気に大規模
5. 数字をビットとして表示
6. データ構造の工夫、PQよりインデックス、位置情報を表示するリストなど

# 課題１　

+ 1）

ヒューリスティック関数を使わないＡ＊ == BFS

１００個のランダムに作成したパズルを解く実行時間：約0.37秒（io出力なし）

![image-20211024112707584](../%E5%9B%BE%E7%89%87/%E5%9B%BE%E5%BA%8A/image-20211024112707584.png)

パズルごとに、正しい状態（＝ゴール状態）から、一歩ランダムな方向にに空タイルを移動することを三十回繰り返す。つまりランダム移転数：３０。

サンプル：

<img src="../%E5%9B%BE%E7%89%87/%E5%9B%BE%E5%BA%8A/image-20211024094246814.png" alt="image-20211024094246814" style="zoom:33%;" />

+ 2）

ゴールの位置にないタイルの数をヒューリスティックとする

１）と同じ条件として、１００個のランダムに作成したパズルを解く実行時間：0.04秒ほど（io出力なし）

たまに0.1秒が出る時がある。

十回：0.161 0.048 0.041 0.044 0.041 0.039 0.045 0.039 0.038 0.042

<img src="../%E5%9B%BE%E7%89%87/%E5%9B%BE%E5%BA%8A/image-20211024112800000.png" alt="image-20211024112800000" style="zoom:50%;" />

+ 3)

マンハッタン距離をヒューリスティックとする

１）と同じ条件として、１００個のランダムに作成したパズルを解く実行時間：0.03秒ほど（io出力なし）

十回：0.033 0.031 0.029 0.027 0.031 0.029 0.032 0.045 0.030 0.027

<img src="../%E5%9B%BE%E7%89%87/%E5%9B%BE%E5%BA%8A/image-20211024113035706.png" alt="image-20211024113035706" style="zoom:50%;" />

# 課題１A:

十万級

<img src="../%E5%9B%BE%E7%89%87/%E5%9B%BE%E5%BA%8A/image-20211024122509673.png" alt="image-20211024122509673" style="zoom:80%;" />

inremd+operator ordering

![image-20211103225242748](../%E5%9B%BE%E7%89%87/%E5%9B%BE%E5%BA%8A/image-20211103225242748.png)

## 8puzzle ソースコード

```c++
#include<iostream>
#include<math.h>
#include<vector>
#include<memory>
#include<time.h>
#include<cstdlib>
#include<queue>
#define block_length 3
using std::cout;
using std::vector;
using std::unique_ptr;
typedef vector<vector<int>> Board;
typedef vector<vector<int>> *ptr_to_b;
int hamming(Board board);
int manhattan(Board board);
void print_board(Board board);


class Node{
    public:
    int moves;
    int prior;
    Board my_board;
    Node *my_parent;
    Node(Board board){
        my_board=board;
        my_parent=nullptr;
        moves=0;
        prior=moves+hamming(board);
    }
    Node(Board board,Node *parent){
        my_board=board;
        my_parent=parent;
        moves=my_parent->moves+1;
        prior=moves+hamming(board);
    }
    ~Node(){
    }
};
struct compare
{
    bool operator() (Node* a, Node* b)
    {
        return a->prior > b->prior; //minimum heap
        //h(s)=0の場合priorの代わりにmovesを使う
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
                int right_x = (temp-1)/block_length;
                int right_y = (temp-1)%block_length;
                manhattan+=abs(i-right_x)+abs(j-right_y);
            }
        }
    }
    return manhattan;
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

Board exchange(ptr_to_b p,int i,int j,int ii,int jj)
{
    Board new_board=*p;
    int temp=new_board[i][j];
    new_board[i][j]=new_board[ii][jj];
    new_board[ii][jj]=temp;
    return new_board;
}

void next_step(Node *current);
void next_step(Node *current){
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
                if (i > 0) { //up
                    new_board = exchange(&board, i,j,i-1,j);
                    if (!has_parent || has_parent && new_board!=parent){
                        Queue.push(new Node(new_board,current));
                    }
                }
                if (i < block_length - 1) { //down
                    new_board = exchange(&board, i, j, i + 1, j);
                    if (!has_parent || has_parent && new_board!=parent){
                        Queue.push(new Node(new_board,current));
                    }
                }
                if (j > 0) { //left
                    new_board = exchange(&board, i, j, i, j - 1);
                    if (!has_parent || has_parent && new_board!=parent){
                        Queue.push(new Node(new_board,current));
                    }
                }
                if (j < block_length - 1) { //right
                    new_board = exchange(&board, i, j, i, j + 1);
                    if (!has_parent || has_parent && new_board!=parent){
                        Queue.push(new Node(new_board,current));
                    }
                }
            }
        }
    }
}

Board random_initial();
Board random_initial(int random_steps){
    int start[block_length][block_length];
    int init[9]={1,2,3,4,5,6,7,8,0};
    int pos_x = 2;
    int pos_y = 2;
    int i,j,random_num;
    for (i=0;i<block_length;i++){
        for (j=0;j<block_length;j++){
            start[i][j]=init[block_length*i+j];
        }
    }
    //0:up,1:down,2:left,3:right
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
        random_num=std::rand() % 4;
        //            cout<<"随机数："<<random_num<<"\n";
        if (pos_x>0 && random_num==0){
            new_board = exchange(&new_board, pos_x,pos_y,pos_x-1,pos_y);
            pos_x--;
        }
        else if (pos_x<block_length-1 && random_num==1){
            new_board = exchange(&new_board, pos_x,pos_y,pos_x+1,pos_y);
            pos_x++;
        }
        else if (pos_y>0 && random_num==2){
            new_board = exchange(&new_board, pos_x, pos_y, pos_x, pos_y - 1);
            pos_y--;
        }
        else if (pos_y<block_length-1 && random_num==3){
            new_board = exchange(&new_board, pos_x, pos_y, pos_x, pos_y + 1);
            pos_y++;
        }
        else{
            continue;
        }

    }
    return new_board;
}

void print_path(Node n);
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
    int loop_times=100;
    Board target=random_initial(0);
    while (loop_times--){
        std::srand(loop_times);
        Board start=random_initial(30);//移动步数
        //        print_board(start);
        Node b1(start);
        next_step(&b1);
        Node* temp=Queue.top();
        while (temp->my_board != target){
            Queue.pop();
            next_step(temp);
            temp=Queue.top();
        }
        //            cout<<"find result!moves:"<<temp->moves<<"\n";
        //            print_path(*temp);
        while (!Queue.empty())
        {
            Queue.pop();
        }
    }
    return 0;
}
```

## 15puzzle ソースコード

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
int hamming(Board board);
int manhattan(Board board);
void print_board(Board board);


class Node{
    public:
    int moves;
    int prior;
    Board my_board;
    Node *my_parent;
    Node(Board board){
        my_board=board;
        my_parent=nullptr;
        moves=0;
        prior=moves+manhattan(board);
    }
    Node(Board board,Node *parent){
        my_board=board;
        my_parent=parent;
        moves=my_parent->moves+1;
        prior=moves+manhattan(board);
    }
    ~Node(){
    }
};
struct compare
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
                int right_x = (temp-1)/block_length;
                int right_y = (temp-1)%block_length;
                manhattan+=abs(i-right_x)+abs(j-right_y);
            }
        }
    }
    return manhattan;
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

Board exchange(ptr_to_b p,int i,int j,int ii,int jj)
{
    Board new_board=*p;
    int temp=new_board[i][j];
    new_board[i][j]=new_board[ii][jj];
    new_board[ii][jj]=temp;
    return new_board;
}

void next_step(Node *current, int *counter);
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
                if (i > 0) { //up
                    new_board = exchange(&board, i,j,i-1,j);
                    if (!has_parent || has_parent && new_board!=parent){
                        Queue.push(new Node(new_board,current));
                        *counter+=1;
                    }
                }
                if (i < block_length - 1) { //down
                    new_board = exchange(&board, i, j, i + 1, j);
                    if (!has_parent || has_parent && new_board!=parent){
                        Queue.push(new Node(new_board,current));
                        *counter+=1;
                    }
                }
                if (j > 0) { //left
                    new_board = exchange(&board, i, j, i, j - 1);
                    if (!has_parent || has_parent && new_board!=parent){
                        Queue.push(new Node(new_board,current));
                        *counter+=1;
                    }
                }
                if (j < block_length - 1) { //right
                    new_board = exchange(&board, i, j, i, j + 1);
                    if (!has_parent || has_parent && new_board!=parent){
                        Queue.push(new Node(new_board,current));
                        *counter+=1;
                    }
                }
            }
        }
    }
}

Board random_initial();
Board random_initial(int random_steps){
    int start[block_length][block_length];
    int init[16]={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0};
    int pos_x = 3;
    int pos_y = 3;
    int i,j,random_num;
    for (i=0;i<block_length;i++){
        for (j=0;j<block_length;j++){
            start[i][j]=init[block_length*i+j];
        }
    }
    //0:up,1:down,2:left,3:right
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
        random_num=std::rand() % 4;
        //            cout<<"随机数："<<random_num<<"\n";
        if (pos_x>0 && random_num==0){
            new_board = exchange(&new_board, pos_x,pos_y,pos_x-1,pos_y);
            pos_x--;
        }
        else if (pos_x<block_length-1 && random_num==1){
            new_board = exchange(&new_board, pos_x,pos_y,pos_x+1,pos_y);
            pos_x++;
        }
        else if (pos_y>0 && random_num==2){
            new_board = exchange(&new_board, pos_x, pos_y, pos_x, pos_y - 1);
            pos_y--;
        }
        else if (pos_y<block_length-1 && random_num==3){
            new_board = exchange(&new_board, pos_x, pos_y, pos_x, pos_y + 1);
            pos_y++;
        }
        else{
            continue;
        }

    }
    return new_board;
}

void print_path(Node n);
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
    Board target=random_initial(0);
    while (loop_times--){
        std::srand(loop_times);
        Board start=random_initial(30);//moving steps
        //        print_board(start);
        Node b1(start);
        next_step(&b1, &count_node);
        Node* temp=Queue.top();
        while (temp->my_board != target){
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

## incremd

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



