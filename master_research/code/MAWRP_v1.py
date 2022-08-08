"""
version 2.1
初步完成平衡聚类算法
第二阶段找临界点的方式可优化
"""
from package import *
from WRP_solver import WatchmanRouteProblem


class State:
    def __init__(self, watchmen):
        self.watchmen = watchmen
        self.A_star_v_l2 = 0  # higher level

    def is_all_finish(self):
        return all([w.is_finish() for w in self.watchmen])

    def get_paths(self):
        return [w.path for w in self.watchmen]


class Watchman:
    def __init__(self):
        self.start = None
        self.empty_cells = set()
        self.edge_list = {}


class MWRP:
    """
    map: 1 means obstacle, 0 means empty cell
    start: start position
    cur_location: the position of current agent,
    seen:cells have been seen so far by the agent, start with seen = LOS(start)
    Expansion: perform all legal movement on location
    LOS: all cells every cell can see
    APSP(All Pairs Shortest Path): the minimum distance between every two cell
        exp: distance between (1,3) and (2,6) = 5 then APSP[(1*self.w+3,2*self.w+6)] = 5
                    be sure that (1,3) is first then (2,6), left top is prior
    empty_cells: all empty cells
    DF_factor: for BJF option, 0 means close the option
    """

    def __init__(self, map_, start, **params):
        self.params = params
        self.map = map_
        self.h, self.w = len(self.map), len(self.map[0])
        self.n_agent = params.get("n_agent")
        self.watchmen = [Watchman() for i in range(self.n_agent)]
        self.empty_cells = set()
        self.start = start
        self.LOS, self.APSP, self.APSP_d = {}, {}, {}
        self.edge_list, self.nodes = {}, 0
        self.pq_l2 = PriorityQueue()

    def run(self):
        start = time.perf_counter()
        self.initialize()
        initial_time = time.perf_counter() - start
        print("initialization complete! time:{} s,".format(initial_time))
        class_ = self.clustering()
        paths = []
        self.visualize([], class_=class_)
        self.watchmen_init(class_)
        for w in self.watchmen:
            temp_params = self.params
            temp_params['empty_cells'] = w.empty_cells
            temp_params['edge_list'] = w.edge_list
            sol = WatchmanRouteProblem(self.map, w.start, **temp_params)
            result = sol.run()
            paths.append(result)
        self.visualize(paths, class_=class_)
        # cur_state = State(self.watchmen)
        # while not cur_state.is_all_finish():  # 中止条件不对,改为一定时间内如果2-layer A* 没有提升就停止
        #     # self.visualize(cur_state.path)
        #     self.next_step(cur_state)
        #     cur_state = self.pq_l2.pop_()
        # self.visualize(cur_state.get_paths(), class_=class_)
        # assert self.check_finish(cur_state.get_paths()), "wrong answer！"
        end_time = time.perf_counter()
        print("total time:{}s,path finding time {}s, expanding nodes:{}"
              .format(end_time - start,
                      end_time - initial_time,
                      self.nodes))

    def watchmen_init(self, class_):
        """
        class_: classification result for cells, class_[0] is not representing for No0.cell,
        but the first non-obstacle cell.
        """
        n_class = self.n_agent
        selected = [0 for i in range(self.n_agent)]  # for designating start point
        real_cell_idx = sorted(list(self.empty_cells))  # after removing obstacles, the idx have changed
        for cur_class in range(n_class):
            class_idx = np.where(class_ == cur_class)[0]  # 外面包了一层元组类型,需要去掉
            # 待调整
            temp_watchman = self.watchmen[cur_class]
            temp_empty_cells = set(real_cell_idx[idx] for idx in class_idx)  # convert to set type
            temp_watchman.empty_cells = temp_empty_cells
            # edge_list
            temp_edge_list = dict()
            for key, value in self.edge_list.items():
                if key in temp_empty_cells:
                    temp_edge_list[key] = self.edge_list[key]
                    # remove the cells not in this watchman
                    temp_edge_list[key] = list(set(temp_edge_list[key]) & temp_empty_cells)
            temp_watchman.edge_list = temp_edge_list
            # start point
            samples = np.random.choice(list(temp_empty_cells), len(temp_empty_cells)//3)
            min_idx = -1  # which start point is the closest to the cluster
            avg_d = float('inf')  # calc average distance between start point and cluster
            for n in range(self.n_agent):
                if selected[n] == 1:
                    continue
                start_code = self.encode(self.start[n][0], self.start[n][1])
                distance_to_start = [self.get_APSP(start_code, c, distance=True) for c in samples]
                new_avg_d = np.average(distance_to_start)
                if new_avg_d < avg_d:
                    avg_d = new_avg_d
                    min_idx = n
            temp_watchman.start = self.start[min_idx]
            selected[min_idx] = 1

    # def synchro(self, watchmen):  # 不考虑每走一步都同步，2-layer search足够
    #     seen = set()
    #     for w in watchmen:
    #         seen = seen | w.seen
    #     for w in watchmen:
    #         w.add_seen(seen)

    def is_obstacle(self, code):
        x, y = self.decode(code)
        return self.map[x, y] == 1

    def clustering(self):
        matrix = np.zeros((self.h*self.w, self.h*self.w+2))
        for pos, d in self.APSP_d.items():
            matrix[[pos[0], pos[1]], [pos[1], pos[0]]] = d  # np.log1p(d)
        # matrix[:, -1] = np.random.randint(0, self.h*self.w//2, size=self.h*self.w)  如果给需要给特征矩阵多加一个随机特征的话

        non_zero_idx = np.where(matrix.sum(axis=1) != 0)[0]  # np.where return tuple type, need [0] to get result
        # if non_zero_idx[0]=5, means the first non-obstacle cell is No.5 cell
        matrix[:, -2:] = np.array([self.decode(i) for i in range(self.h*self.w)])
        matrix = matrix[non_zero_idx, :]

        # result = get_even_clusters(matrix, n_clusters=self.n_agent)  # bad performance
        # result = KMeans(self.n_agent).fit_predict(matrix)
        result = MyKmeans(self.n_agent, non_zero_idx, self.APSP_d, self.edge_list).fit_predict(matrix)
        print("clustering result:", Counter(result))
        # result = self.balanced_connect_processing(result, non_zero_idx)
        # result = self.MYKmeans(matrix, non_zero_idx)
        return np.array(result)

    def check_finish(self, paths):
        map_ = self.map.copy()
        axis_x = []
        axis_y = []
        for path in paths:
            for pos in path:
                sights = self.LOS[pos]
                for sight in sights:
                    x, y = self.decode(sight)
                    axis_x.append(x)
                    axis_y.append(y)
            map_[axis_x, axis_y] = 1
        return map_.sum() == self.w*self.h

    def next_step(self, cur_state):
        """
        每个watchman都要准备好下一步，推入pq_l1,pq_l1的队中元素是watchman class
        """
        _, near_watchers = self.make_graph(cur_state.seen, cur_state.path,
                                           IW=self.IW, WR=self.WR, BJP_DF=self.DF_factor)
        for new_pos in near_watchers:
            new_x, new_y = self.decode(new_pos)
            path = deepcopy(self.get_APSP(cur_state.cur_pos, new_pos, distance=False))  # the path will go through
            # print(f"cur_pos:{a} to new_pos:{b}: {path}")
            assert 0 <= new_x < self.h and 0 <= new_y < self.w and self.map[new_x][new_y] != 1
            temp_state = deepcopy(cur_state)
            cur_path = temp_state.path
            if path[-1] != new_pos:
                path = path[::-1]
            path.remove(cur_state.cur_pos)
            cur_path.extend(path)
            cur_seen = temp_state.seen
            for cell in path:
                cur_seen = cur_seen | self.LOS[cell]
            h_value = self.calc_MST_h(cur_seen, cur_path)
            # h_value = self.calc_TSP_h(cur_seen, cur_path)
            # h_value = 0
            # h_value = self.calc_agg_h(cur_seen, cur_path)
            # print("h_value:", h_value)
            A_star_v = self.calc_A_stat_v(len(cur_path), h_value, w=self.f_weight, option=self.f_option)
            self.pq.push_(State(cur_path, cur_seen, A_star_v))
            self.nodes += 1

    def calc_A_stat_v(self, g, h, w=1, option="WA"):
        if option == "WA":
            return g+w*h
        elif option == "XDP":
            return 1/(2*w)*(g+(2*w-1)*h+math.sqrt((g-h)**2+4*w*g*h))
        elif option == "XUP":
            return 1/(2*w)*(g+h+math.sqrt((g+h)**2+4*w*(w-1)*h*h))

    def encode(self, x, y):
        return x * self.w + y

    def decode(self, code):
        return code // self.w, code % self.w

    def get_APSP(self, a, b, distance=True):
        if a > b:
            a, b = b, a
        elif a == b:
            return 0
        return self.APSP_d[(a, b)] if distance else self.APSP[(a, b)]

    def initialize(self):
        """
        prepare two lookup tables for efficiency
        self.LOS, self.empty_cells, self.APSP, self.APSP_d, self.edge_list
        """
        adjacent_matrix = np.zeros((self.h * self.w, self.h * self.w))
        for x in range(self.h):
            for y in range(self.w):
                if self.map[x, y] == 0:
                    code = self.encode(x, y)
                    self.empty_cells.add(code)
                    self.LOS[code] = LOS4(self.map, code)
                    # for num in range(code + 1, self.h * self.w):  # 与其他所有点的最短距离
                    #     cell_x, cell_y = self.decode(num)
                    #     if self.map[cell_x, cell_y] == 1:
                    #         continue
                    #     path, d = self.minimum_d((x, y), (cell_x, cell_y))
                    #     a, b = code, num
                    #     # make sure a is smaller
                    #     assert a < b
                    #     self.APSP_d[(a, b)] = d
                    #     self.APSP[(a, b)] = path
                    # build edge list and adjacent matrix
                    for new_x, new_y in [(x + 1, y), (x, y + 1)]:  # 图会自动建立双向边,所以只需要向前考虑
                        if 0 <= new_x < self.h and 0 <= new_y < self.w and self.map[new_x][new_y] != 1:
                            start = self.encode(x, y)
                            end = self.encode(new_x, new_y)
                            if start not in self.edge_list:
                                self.edge_list[start] = [end]
                            else:
                                self.edge_list[start].append(end)
                            if end not in self.edge_list:
                                self.edge_list[end] = [start]
                            else:
                                self.edge_list[end].append(start)
                            # build adjacent matrix
                            adjacent_matrix[[start, end], [end, start]] = 1
        # build APSP and APSP_d
        graph = csr_matrix(adjacent_matrix)
        dist_matrix, predecessors = floyd_warshall(csgraph=graph, directed=False, return_predecessors=True)
        temp_list = list(self.empty_cells)
        temp_list.sort()
        for i in range(len(temp_list)):
            a = temp_list[i]
            for j in range(i+1, len(temp_list)):
                b = temp_list[j]
                self.APSP[(a, b)] = self.get_path_from_pred(predecessors, a, b)
                self.APSP_d[(a, b)] = int(dist_matrix[a, b])
        del temp_list, dist_matrix, predecessors
        # check legality
        for i in range(self.n_agent):
            x, y = self.start[i]
            assert self.map[x, y] == 0, "出发点不符合条件"
        # initialize
        self.params["LOS"] = self.LOS
        self.params["empty_cells"] = self.empty_cells
        self.params["APSP"] = self.APSP
        self.params["APSP_d"] = self.APSP_d
        self.params["edge_list"] = self.edge_list

    def get_path_from_pred(self, pred_m, start, end):
        path = [end]
        pred = pred_m[start, end]
        while pred != start:
            path.append(pred)
            pred = pred_m[start, pred]
        path.append(start)
        return path[::-1]

    def minimum_d(self, start, end):
        """
        calc distance and route between two points a and b
        breadth first search
        """
        start_x, start_y = start
        target_x, target_y = end
        if self.map[start_x, start_y] == 1 or \
                self.map[target_x, target_y] == 1:
            print("illegal cell!")
            return
        queue = [(start_x, start_y, 0)]
        current_map = self.map.copy()
        q = deque(queue)
        pre_step = {}  # record the last step pre_step[5]=8 means path 8->5
        while q:
            x, y, d = q.popleft()
            # print(f"{a},{b} distance", (x, y, d))
            if (x, y) == (target_x, target_y):  # find the target, record path
                # print(f"{a},{b} distance", (x, y, d))
                min_path = []
                current = self.encode(target_x, target_y)
                while current != self.encode(start_x, start_y):
                    min_path.append(current)
                    current = pre_step[current]
                min_path.append(self.encode(start_x, start_y))
                return min_path[::-1], d
            for new_x, new_y in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                if 0 <= new_x < self.h and 0 <= new_y < self.w and current_map[new_x, new_y] != 1:
                    current_map[new_x, new_y] = 1
                    q.append((new_x, new_y, d + 1))
                    pre_step[self.encode(new_x, new_y)] = self.encode(x, y)
        return

    def plot_lines(self, mat_w, mat_h):
        left_border = up_border = -0.5
        right_border, down_border = mat_w-0.5, mat_h-0.5
        plt.hlines([i-0.5 for i in range(mat_h)], left_border, right_border, color='black')
        plt.vlines([i-0.5 for i in range(mat_w)], up_border, down_border, color='black')

    def visualize(self, paths, class_):
        """
        see where the cell is
        """
        plt.matshow(-self.map, cmap=plt.cm.hot)
        mat_w, mat_h = len(self.map[0]), len(self.map)
        self.plot_lines(mat_w, mat_h)

        # plot cluster
        axis_x, axis_y = [], []
        for i in sorted(list(self.empty_cells)):  # have deleted the obstacle cells
            x, y = self.decode(i)
            axis_x.append(x)
            axis_y.append(y)
        plt.scatter(axis_y, axis_x, c=class_, s=100, alpha=0.4)  # attention! x and y are reversed!

        # plot path for every agent
        for i, path in enumerate(paths):
            color = plt.cm.Set1(i)
            length = len(path)
            start_x, start_y = self.decode(path[0])
            # print(f"start:{start_x},{start_y}")
            plt.text(start_y, start_x, s=f'S{i}', fontsize='x-large', ha='center', va='center',
                     color=color)

            for j in range(length - 1):
                x_1, y_1 = self.decode(path[j])
                x_2, y_2 = self.decode(path[j+1])
                dx_ = x_2 - x_1
                dy_ = y_2 - y_1
                plt.arrow(y_1, x_1, dx=dy_, dy=dx_, width=0.01, ec=color, alpha=1,
                          fc=color,
                          head_width=0.2,
                          length_includes_head=True)  # 坐标系位置和矩阵cell位置表示是相反的
        plt.show()


def read_map(path):
    matrix = []
    with open(path) as f:
        for row, line in enumerate(f.readlines()[4:]):  # 前3行无用
            line = line.strip('\n')
            temp = []
            for col, alpha in enumerate(line):
                if alpha == '.':
                    temp.append(0)
                else:
                    temp.append(1)
            matrix.append(temp)
    return np.array(matrix)


def main():
    path = r"..\maps"
    files = os.listdir(path)
    os.chdir(path)
    params = {"f_weight": 1, "f_option": "WA",
              "DF_factor": 2, "IW": True, "WR": False,
              "n_agent": 3, "heuristic":  "MST"}    # TSP,MST,agg_h, None
    # start = None  # give the pos responding to n_agent
    start = [(0, 0), (20, 0), (20, 20)]
    # map = np.array([[1, 0, 0, 0],
    #                 [1, 0, 1, 1],
    #                 [0, 0, 0, 0],
    #                 [0, 1, 1, 0],
    #                 [0, 0, 0, 0]])
    for file in files:
        print(file)
        map = read_map(file)
        sol = MWRP(map, start, **params)
        sol.run()
        break


main()
