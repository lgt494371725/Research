#include "exchange.h"
#include "memory_pool.hpp"

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
	void* operator new(size_t size);
	void operator delete(void* p);
	~Node(){};
};

MemoryPool<sizeof(Node), 1000> gMemPool;

void* Node::operator new(size_t size)
{
	//std::cout << "new object space" << std::endl;
	return gMemPool.allocate();
}

void Node::operator delete(void* p)
{
	//std::cout << "free object space" << std::endl;
	gMemPool.deallocate(p);
}

Node random_initial(int random_steps);
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
                if (j < block_length - 1) { //right
                    new_board = exchange(&board, i, j, i, j + 1);
                    if (!has_parent || has_parent && new_board!=parent){
                        position new_pos4 = cur_pos;
                        new_pos4[1]++;
                        Queue.push(new Node(new_board,current,new_pos4));
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


                if (i < block_length - 1) { //down
                    new_board = exchange(&board, i, j, i + 1, j);
                    if (!has_parent || has_parent && new_board!=parent){
                        position new_pos2 = cur_pos;
                        new_pos2[0]++;
                        Queue.push(new Node(new_board,current,new_pos2));
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
//            delete temp;
            temp=Queue.top();
        }
////            cout<<"find result!moves:"<<temp->moves<<"\n";
////            print_path(*temp);
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
