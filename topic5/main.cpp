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

Node initial(int is_start);
void print_path(Node n);

struct compare //重写仿函数
{
    bool operator() (Node* a, Node* b)
    {
        return a->prior > b->prior; //minimum heap
    }
};

std::priority_queue<Node*,std::vector<Node*>, compare> Queue;

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
    int right_x = value%block_length;
    int right_y = value/block_length;
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

Node initial(int is_start){
    int start[block_length][block_length];
    int init[9];
    position pos;
	if (is_start == 1){
	int temp[9]={0,3,6,1,4,7,2,5,8};
    for (int i = 0; i < 9; i++)
        init[i]=temp[i];
    pos.push_back(0);
    pos.push_back(0);
    }
    else if (is_start == 2){  // puzzle 1
    int temp[9]={8,0,2,7,4,5,6,1,3};
    for (int i = 0; i < 9; i++)
        init[i]=temp[i];
    pos.push_back(0);
    pos.push_back(1);	
	}
	else if (is_start == 3){ // puzzle 2
    int temp[9]={8,5,2,0,4,3,6,7,1};
    for (int i = 0; i < 9; i++)
        init[i]=temp[i];
    pos.push_back(1);
    pos.push_back(0);	
	} 
	else if (is_start == 4){ // puzzle 3
    int temp[9]={8,7,4,5,2,1,6,3,0};
    for (int i = 0; i < 9; i++)
        init[i]=temp[i];
    pos.push_back(2);
    pos.push_back(2);	
	} 
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
    int loop_times=1;
    double dur;
    clock_t start,end;
    start=clock();
    Node target=initial(1);
    while (loop_times--){
        std::srand(loop_times);
        Node start=initial(4);//移动步数
        next_step(&start, &count_node);
    	Node* temp=Queue.top();
        while (temp->my_board != target.my_board){
            Queue.pop();
            next_step(temp, &count_node);
//            delete temp;
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
