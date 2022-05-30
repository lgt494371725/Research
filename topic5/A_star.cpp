    #include<iostream>
    #include<math.h>
    #include<vector>
    #include<memory>
    #include<time.h>
    #include<cstdlib>
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

    vector<Node*> Queue;

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
                    int right_x = temp%block_length;
                    int right_y = temp/block_length;
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
//            cout<<"父亲的父亲！\n";
//            print_board(parent);
            has_parent=true;
        }
//        cout<<"parent:\n";
//        print_board(current->my_board);
//        cout<<"\n";
//        cout<<"可能路径:\n";
        Board new_board;
        for (int i=0;i<block_length;i++){
            for (int j=0;j<block_length;j++){
                if (board[i][j]==0){
                    if (i > 0) { //up
                        new_board = exchange(&board, i,j,i-1,j);
                        if (!has_parent || has_parent && new_board!=parent){
    //						unique_ptr<Node> temp1(new Node(new_board,current));
                            Queue.insert(Queue.begin(),new Node(new_board,current));
    //						Queue.insert(Queue.begin(),temp1);
    //	                    print_board(new_board);
                        }
                    }
                    if (i < block_length - 1) { //down
                        new_board = exchange(&board, i, j, i + 1, j);
                        if (!has_parent || has_parent && new_board!=parent){
    //						unique_ptr<Node> temp2(new Node(new_board,current));
    //						Queue.insert(Queue.begin(),temp2);
                            Queue.insert(Queue.begin(),new Node(new_board,current));
    //	                    print_board(temp2->my_board);
    //                        print_board(temp2->my_parent->my_board);
                        }
                    }
                    if (j > 0) { //left
                        new_board = exchange(&board, i, j, i, j - 1);
                        if (!has_parent || has_parent && new_board!=parent){
    //						unique_ptr<Node> temp3(new Node(new_board,current));
    //						Queue.insert(Queue.begin(),temp3);
                            Queue.insert(Queue.begin(),new Node(new_board,current));
    //	                    print_board(temp3->my_board);
    //                        print_board(temp3->my_parent->my_board);
                        }
                    }
                    if (j < block_length - 1) { //right
                        new_board = exchange(&board, i, j, i, j + 1);
                        if (!has_parent || has_parent && new_board!=parent){
    //						unique_ptr<Node> temp4(new Node(new_board,current));
    //						Queue.insert(Queue.begin(),temp4);
    //	                    print_board(new_board);
                            Queue.insert(Queue.begin(),new Node(new_board,current));
                        }
                    }
                }
            }
        }
    }

    Board random_initial();
    Board random_initial(int random_steps){
        int start[block_length][block_length];
        int init[9]={0,3,6,1,4,7,2,5,8};
        int pos_x = 0;
        int pos_y = 0;
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
        cout<<"打印路径：\n";
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
        //	for (int i=0;i<Queue.size();i++)
        //	{
        //		print_board(Queue[i].my_board);
        //		cout<<"\n";
        //	}
            while (Queue.size() != 0){
                Node* temp=Queue.back();
    //            cout<<"取出:\n";
    //            print_board(temp->my_board);
    //            cout<<"\n";
                if (temp->my_board == target){
                    cout<<"find result!moves:"<<temp->moves<<"\n";
                    print_path(*temp);
                    break;
                }
                next_step(temp);
                Queue.pop_back();
            }
            Queue.clear();
        }
        return 0;
    }

