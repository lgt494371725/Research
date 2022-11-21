"""
version 4.4
profiling进行效率优化
定义实验函数
final date: 2022.11.21
"""
import pandas as pd

from package import *
from WRP_solver import WatchmanRouteProblem
from config import optional_params


class State:
    MAX_VAR_V = None

    def __init__(self, watchmen, h, w):
        self.h = h
        self.w = w
        self.watchmen = {w.get_number(): w for w in watchmen}
        self.f_v = self.calc_f_v(self.get_paths(only_length=True))  # higher level
        self.clustering = None

    def get_paths(self, only_length=False):
        if only_length:
            return {w.get_number(): len(w.path) - 1 for w in self.watchmen.values()}
        else:
            return {w.get_number(): w.path for w in self.watchmen.values()}

    def get_watchman(self, number):
        assert number in self.watchmen.keys(), "watchman not found".center(30, '=')
        return self.watchmen[number]

    def update_watchman(self, new_watchman):
        number = new_watchman.get_number()
        assert number in self.watchmen.keys(), "watchman not found".center(30, '=')
        self.watchmen[number] = new_watchman
        self.f_v = self.calc_f_v(self.get_paths(only_length=True))

    @classmethod
    def calc_f_v(cls, paths: dict):
        paths = list(paths.values())
        var_v, max_v = np.var(paths), max(paths)
        w = var_v / cls.MAX_VAR_V
        f_v = round(w * var_v) + max(paths) + round(np.mean(paths))
        return f_v

    def get_class(self, no_obstacles=False):
        """
        :param no_obstacles:  if true, remove cells which are obstacles
        """
        n_agent = len(self.watchmen)
        clustering = np.zeros((self.h * self.w, n_agent))
        for num, w in self.watchmen.items():
            clustering[list(w.empty_cells), num] = 1
        if no_obstacles:
            non_zero_cell_idx = np.where(np.sum(clustering, axis=1) != 0)[0]
            clustering = clustering[non_zero_cell_idx]
        return clustering

    def get_value(self):
        return self.f_v


class Watchman:
    def __init__(self, number):
        self.No = number
        self.start = None
        self.empty_cells = set()
        self.edge_list = {}
        self.path = None

    def get_number(self):
        return self.No

    def get_path_len(self):
        if self.path is None:
            print("no path exist!!")
            return
        return len(self.path)


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
        self.watchmen = [Watchman(i) for i in range(self.n_agent)]
        self.empty_cells = set()
        self.start = params.get("start")
        self.LOS = [[] for i in range(self.h * self.w)]
        self.APSP = Myarray((self.h * self.w, self.h * self.w))
        self.APSP_d = np.zeros((self.h * self.w, self.h * self.w))
        self.edge_list, self.nodes = {}, 0
        self.pq_l2 = PriorityQueue()
        self.path_cache = {}

    def restart(self):
        self.start = None
        self.empty_cells = set()
        self.watchmen = [Watchman(i) for i in range(self.n_agent)]
        self.LOS = [[] for i in range(self.h * self.w)]
        self.APSP = Myarray((self.h * self.w, self.h * self.w))
        self.APSP_d = np.zeros((self.h * self.w, self.h * self.w))
        self.edge_list = {}
        self.pq_l2 = PriorityQueue()
        self.path_cache = {}

    def get_path(self, watchman):
        hash_value = self.get_hash_value(watchman.start, watchman.empty_cells)
        if hash_value in self.path_cache:
            path = self.path_cache[hash_value]
            # print("get path from cache!!")
        else:
            path = self.solve_WRP(watchman)
            self.path_cache[hash_value] = path
        return path

    def get_hash_value(self, start, empty_cells):
        start_s = str(self.encode(start[0], start[1]))
        s = start_s + "".join([*map(str, empty_cells)])
        s = s.encode('utf-8')
        hash_v = hashlib.md5(s).hexdigest()
        return hash_v

    def run(self, test_times):
        np.random.seed(42)
        start = time.perf_counter()
        output_file_name = f"{self.params['f_weight']}_{self.n_agent}_{self.params['heuristic']}.txt"
        if self.params['write_to_file']:
            f = open(f"../results/{output_file_name}", "w+")
        max_paths_len = []
        bar = tqdm(range(test_times))
        for i in bar:
            bar.set_description("test time processing")
            try:

                # initial stage
                self.initialize()
                # initial_time = time.perf_counter() - start
                # print("initialization complete! time:{} s,".format(initial_time))
                class_ = self.clustering()
                self.watchmen_init(class_)
                for w in self.watchmen:
                    w.path = self.solve_WRP(w)
                    hash_value = self.get_hash_value(w.start, w.empty_cells)
                    self.path_cache[hash_value] = w.path
                cur_max_p = max([len(i) - 1 for i in self.path_cache.values()])
                MAX_VAR_V = np.var([0] * (self.n_agent - 1) + [cur_max_p])
                State.MAX_VAR_V = MAX_VAR_V
                cur_state = State(self.watchmen, h=self.h, w=self.w)
                if self.params['visualize']:
                    self.visualize(list(cur_state.get_paths().values()), class_=cur_state.get_class(no_obstacles=True))
                best_state = cur_state

                # improve stage
                max_iter = 300
                early_stop = 0
                tolerant = 10
                for loop in range(max_iter):  # loop until result converges
                    self.next_step(cur_state)
                    if self.pq_l2.is_empty():
                        break
                    cur_state = self.pq_l2.pop_()
                    if best_state.f_v >= cur_state.f_v:  # 停止条件需要接着优化
                        best_state = cur_state
                    else:
                        early_stop += 1
                        if early_stop == tolerant:
                            print("result has been stable, early stop!!")
                            break
                    # self.visualize(list(cur_state.get_paths().values()), class_=cur_state.get_class(no_obstacles=True))
                # assert self.check_finish(best_state.get_paths()), "wrong answer！"
                if self.params['visualize']:
                    self.visualize(list(best_state.get_paths().values()),
                                   class_=best_state.get_class(no_obstacles=True))
                # log
                paths_length = list(best_state.get_paths(only_length=True).values())
                max_paths_len.append(max(paths_length) - 1)
                if self.params['write_to_file']:
                    f.write(f"round{i}:start_pos:{self.start}, paths length:{paths_length}, min_sum:{sum(paths_length)}"
                            f", min_max:{max(paths_length)}\n")
                self.restart()

            except Exception as e:
                print(f"----------main program wrong:{e}")
                self.restart()
                continue
        total_time = time.perf_counter() - start
        expansion = self.nodes / test_times
        successful = len(max_paths_len)
        avg_max_paths_len = round(np.mean(max_paths_len), 3)
        paths_len_std = round(np.std(max_paths_len), 3)
        per_exe_time = total_time/successful
        string = "total time:{:.3f}s, expansion:{}, successful time:{}, " \
                 "avg_max_paths_len+std:{:.3f}±{:.3f}, per_exe_time:{}\n".format(total_time, expansion,
                                                                                 len(max_paths_len), avg_max_paths_len,
                                                                                 paths_len_std, per_exe_time)
        if self.params['write_to_file']:
            f.write(string)
            f.close()
        return_dict = {"total_time": total_time, "expansion": expansion,
                       "avg_max_paths_len": avg_max_paths_len, "paths_len_std": paths_len_std}
        return return_dict

    def watchmen_init(self, class_):
        """
        class_: classification result for cells, class_[0] is not representing for No0.cell,
        but the first non-obstacle cell.
        start point assignment is a linear assignment problem
        make a matrix with row: start_pos col: cluster  value:average distance
        """
        n_class = self.n_agent
        selected = [0 for i in range(self.n_agent)]  # for designating start point
        real_cell_idx = sorted(list(self.empty_cells))  # after removing obstacles, the idx have changed
        # start_cluster_m = np.zeros((self.n_agent, self.n_agent))

        for cur_class in range(n_class):
            assert isinstance(class_, np.ndarray)
            class_idx = np.where(class_[:, cur_class] == 1)[0]  # 外面包了一层元组类型,需要去掉
            temp_watchman = self.watchmen[cur_class]
            temp_empty_cells = [real_cell_idx[idx] for idx in class_idx]  # need convert to set type later
            # start point assignment
            min_idx = -1  # which start point is the closest to the cluster
            avg_d = float('inf')  # calc average distance between start point and cluster
            closest_cell = -1
            for n in range(self.n_agent):
                if selected[n] == 1:
                    continue
                start_code = self.encode(self.start[n][0], self.start[n][1])
                distance_to_start = [self.APSP_d[start_code, c] for c in temp_empty_cells]
                new_closest_cell = temp_empty_cells[np.argmin(distance_to_start)]  # used later
                new_avg_d = np.average(distance_to_start)
                if new_avg_d < avg_d:
                    avg_d = new_avg_d
                    min_idx = n
                    closest_cell = new_closest_cell
            cur_start_pos = self.start[min_idx]
            temp_watchman.start = cur_start_pos
            selected[min_idx] = 1
            """
            there is a possibility that start_pos doesn't in the internal of cluster
            so need to find the cell A closest to the start_pos B in the cluster 
            and incorporate all cells on the path between A and B
            """
            temp_empty_cells = set(temp_empty_cells)
            cur_start_pos_code = self.encode(cur_start_pos[0], cur_start_pos[1])
            if cur_start_pos_code in temp_empty_cells:
                temp_watchman.empty_cells = temp_empty_cells
            else:
                path_cells = self.APSP[cur_start_pos_code, closest_cell]
                temp_empty_cells = temp_empty_cells | set(path_cells)
                temp_watchman.empty_cells = temp_empty_cells
            # edge_list
            temp_edge_list = dict()
            for key, value in self.edge_list.items():
                if key in temp_empty_cells:
                    temp_edge_list[key] = self.edge_list[key]
                    # remove the cells not in this watchman
                    temp_edge_list[key] = list(set(temp_edge_list[key]) & temp_empty_cells)
            temp_watchman.edge_list = temp_edge_list

    def is_obstacle(self, code):
        x, y = self.decode(code)
        return self.map[x, y] == 1

    def clustering(self):
        matrix = np.zeros((self.h * self.w, self.h * self.w + 2))
        matrix[:, :self.h * self.w] = self.APSP_d.copy()

        non_zero_idx = np.where(matrix.sum(axis=1) != 0)[0]  # np.where return tuple type, need [0] to get result
        # if non_zero_idx[0]=5, means the first non-obstacle cell is No.5 cell
        matrix[:, -2:] = np.array([self.decode(i) for i in range(self.h * self.w)])

        # add random features
        # random_m = np.random.randint(0, self.h*self.w//2, size=(self.h*self.w, self.h*self.w//4))
        # matrix = np.hstack((matrix, random_m))

        # remove obstacle cell_idx
        matrix = matrix[non_zero_idx, :]
        # result = KMeans(self.n_agent).fit_predict(matrix)
        n_start_pos = [self.encode(pos[0], pos[1]) for pos in self.start]
        result = MyKmeans2(self.n_agent, non_zero_idx, self.APSP_d, self.edge_list, n_start_pos).fit_predict(matrix)
        counter = {}
        for col in range(result.shape[1]):
            counter[col] = sum(result[:, col] != 0)
        print(f"start pos: {self.start} \nclustering result:{counter}", )
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
        return map_.sum() == self.w * self.h

    def next_step(self, cur_state):
        """
        每个watchman都要准备好下一步，推入pq_l1,pq_l1的队中元素是watchman class
        assigned: 一个step中, 如果某个cell从cls_1到cls_2的分配已经发生过，就不应该有下一次
        """

        def is_next_1(need_to_assigns, new_path_cls_1):
            succ = need_to_assigns[1]
            for c in new_path_cls_1:
                if self.APSP_d[succ, c] <= 1:  # find successor
                    # self.show_real_pos([succ, c])
                    return True
            return False

        def is_next_2(need_to_assigns, watchman_1):
            # there may be some outliers
            left_cells = watchman_1.empty_cells - set(need_to_assigns)
            left_cells = {self.decode(i): 1 for i in left_cells}
            left_cells = depth_first_search(self.map, left_cells, watchman_1.start)
            left_cells = set([self.encode(x, y) for x, y in left_cells])
            need_to_assigns = watchman_1.empty_cells - left_cells
            # self.show_real_pos(need_to_assigns)

            # except cell on the path
            for c in need_to_assigns - {closest_idx}:
                for r in left_cells:
                    if self.APSP_d[r, c] <= 1:
                        return True
            return False

        tolerance_size = 0
        clustering = cur_state.get_class()
        assigned = {}
        for cell, candidate in self.edge_list.items():  # plan to assign cell from cls_1 to cls_2
            cls_1s = np.where(clustering[cell] != 0)[0]  # possibly multi-class
            for cls_1 in cls_1s:
                watchman_1 = cur_state.get_watchman(cls_1)
                path_len_cls_1 = watchman_1.get_path_len()
                for nxt_cell in candidate:
                    cls_2s = np.where(clustering[nxt_cell] != 0)[0]
                    for cls_2 in cls_2s:
                        assign_code = f"{cell}_{cls_1}_{cls_2}"
                        if assigned.get(assign_code, 0):  # if exists, skip
                            continue
                        # print(f"cell:{cell}， class:{cls_1}", f"|nxt_cell:{nxt_cell},class:{cls_2}")
                        assigned[assign_code] = 1
                        watchman_2 = cur_state.get_watchman(cls_2)
                        path_len_cls_2 = watchman_2.get_path_len()
                        # do assignment only when path of cls_1 longer than path of cls_2
                        if cls_1 == cls_2 or (path_len_cls_1 - path_len_cls_2 < tolerance_size):
                            continue
                        """
                        from different watchman and length demand satisfied, do assign operation
                        find the closest cell in the path
                        """

                        path_cls_1 = watchman_1.path
                        distances = [self.APSP_d[cell, c] for c in path_cls_1]
                        closest_idx = path_cls_1[np.argmin(distances)]
                        # if start point, skip
                        if closest_idx == self.encode(watchman_1.start[0], watchman_1.start[1]):
                            continue
                        # self.show_real_pos([cell, nxt_cell])
                        path_to_closest_idx = set(self.APSP[cell, closest_idx])
                        # 考虑分配这个点会影响到的所有点
                        cell_idx_in_path = np.where(np.array(path_cls_1) == closest_idx)[0]
                        assert isinstance(path_cls_1, list)
                        if len(cell_idx_in_path) == 1:  # appears only once in the route
                            need_to_assigns = path_cls_1[cell_idx_in_path[0]:]
                            new_path_cls_1 = path_cls_1[:cell_idx_in_path[0]]
                        elif len(cell_idx_in_path) == 2:
                            need_to_assigns = path_cls_1[cell_idx_in_path[0]:cell_idx_in_path[1] + 1]
                            new_path_cls_1 = path_cls_1[:cell_idx_in_path[0]] + \
                                             path_cls_1[cell_idx_in_path[1] + 1:]
                        else:  # too many times, give up assigning
                            continue
                        """
                        确认连通性, 如果存在以下情况，则去掉当前点即可，后续路径保留
                        is_next_1.后续路径与剩余的路径有接点(如走到当前点又回头了)
                        is_next_2.除了当前点外，关联的后续路径以及他们牵连到的cell中，与agent剩余cell
                        能够找到连接点
                        """
                        if len(need_to_assigns) > 1:
                            if is_next_1(need_to_assigns, new_path_cls_1) or \
                                    is_next_2(need_to_assigns, watchman_1):
                                need_to_assigns = [need_to_assigns[0]]
                        """
                        ---------condition2---------------
                        assess the improvement of f_v, if no improvement, skip
                        """
                        epsilon = 1.5
                        pre_f_v = cur_state.f_v
                        post_paths = cur_state.get_paths(only_length=True)
                        post_paths[cls_1] -= len(need_to_assigns)
                        # need to consider the duplication when calc estimated length of cls_2
                        post_paths[cls_2] += len(set(need_to_assigns) - watchman_2.empty_cells)
                        post_f_v = State.calc_f_v(post_paths)

                        if post_f_v / pre_f_v >= epsilon:
                            continue
                        # path_to_closest_idx also need to assign
                        need_to_assigns = set(need_to_assigns)
                        need_to_assigns |= path_to_closest_idx
                        # except cells on the path, there may be some outliers come out
                        left_cells = watchman_1.empty_cells - need_to_assigns

                        left_cells = {self.decode(i): 1 for i in left_cells}
                        new_can_reach = depth_first_search(self.map, left_cells, watchman_1.start)
                        new_can_reach = set([self.encode(x, y) for x, y in new_can_reach])

                        # self.show_real_pos(need_to_assigns)
                        outliers = watchman_1.empty_cells - new_can_reach
                        need_to_assigns |= outliers

                        start_code = self.encode(watchman_1.start[0], watchman_1.start[1])
                        if start_code in need_to_assigns:
                            need_to_assigns -= {start_code}  # start point is not allowed to assign
                        # adjust new watchmen empty cell and edge_list
                        new_watchman_1 = Watchman(watchman_1.No)
                        new_watchman_1.start = watchman_1.start
                        new_watchman_1.empty_cells = watchman_1.empty_cells - need_to_assigns
                        new_watchman_1.edge_list = self.calc_new_edge_list(new_watchman_1.empty_cells)

                        new_watchman_2 = Watchman(watchman_2.No)
                        new_watchman_2.start = watchman_2.start
                        to_cls2_cells = need_to_assigns - watchman_2.empty_cells
                        new_watchman_2.empty_cells = watchman_2.empty_cells | to_cls2_cells
                        if to_cls2_cells == set():
                            new_watchman_2.edge_list = watchman_2.edge_list
                        else:
                            new_watchman_2.edge_list = self.calc_new_edge_list(new_watchman_2.empty_cells)

                        # calc new path len and store it into path_cache
                        new_watchman_1.path = self.get_path(new_watchman_1)
                        new_watchman_2.path = self.get_path(new_watchman_2)

                        # 剩下的步骤是生成新的状态，然后放入队列中
                        new_state = deepcopy(cur_state)
                        new_state.update_watchman(new_watchman_1)
                        new_state.update_watchman(new_watchman_2)
                        # print(f"add to pq, cell:{cell}", f"nxt_cell:{nxt_cell}")
                        # self.visualize(list(new_state.get_paths().values()),
                        #                class_=new_state.get_class(no_obstacles=True))
                        self.pq_l2.push_(new_state)
                        self.nodes += 1
        return None

    def calc_new_edge_list(self, empty_cells):
        temp_edge_list = {}
        for c in empty_cells:
            temp_edge_list[c] = list(set(self.edge_list[c]) & empty_cells)
        return temp_edge_list

    def solve_WRP(self, watchman):
        temp_params = self.params
        temp_params['empty_cells'] = watchman.empty_cells
        temp_params['edge_list'] = watchman.edge_list
        sol = WatchmanRouteProblem(self.map, watchman.start, **temp_params)
        result = sol.run()
        return result

    def calc_h(self, paths) -> int:
        return 0

    def show_real_pos(self, cells):
        real_pos_cells = [self.decode(cell) for cell in cells]
        print(real_pos_cells)

    def calc_A_stat_v(self, g, h, w=1, option="WA"):
        if option == "WA":
            return g + w * h
        elif option == "XDP":
            return 1 / (2 * w) * (g + (2 * w - 1) * h + math.sqrt((g - h) ** 2 + 4 * w * g * h))
        elif option == "XUP":
            return 1 / (2 * w) * (g + h + math.sqrt((g + h) ** 2 + 4 * w * (w - 1) * h * h))

    def encode(self, x, y):
        return x * self.w + y

    def decode(self, code):
        return code // self.w, code % self.w

    def initialize(self):
        """
        prepare two lookup tables for efficiency
        self.LOS, self.empty_cells, self.APSP, self.APSP_d, self.edge_list
        if start_pos not specified, random generation will be applicated
        """
        adjacent_matrix = np.zeros((self.h * self.w, self.h * self.w))
        for x in range(self.h):
            for y in range(self.w):
                if self.map[x, y] == 0:
                    code = self.encode(x, y)
                    self.empty_cells.add(code)
                    self.LOS[code] = LOS4(self.map, code)
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
            for j in range(i + 1, len(temp_list)):
                b = temp_list[j]
                self.APSP[a, b] = self.get_path_from_pred(predecessors, a, b)
                self.APSP[b, a] = self.APSP[a, b]
                self.APSP_d[[a, b], [b, a]] = int(dist_matrix[a, b])
        del temp_list, dist_matrix, predecessors
        # generate start_pos
        if not self.start:
            cand_start = np.random.choice(list(self.empty_cells), self.n_agent, replace=False)
            self.start = [*map(lambda x: self.decode(x), cand_start)]
            # print(f"The start point is randomly generated, {self.start}")
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
        right_border, down_border = mat_w - 0.5, mat_h - 0.5
        plt.hlines([i - 0.5 for i in range(mat_h)], left_border, right_border, color='black')
        plt.vlines([i - 0.5 for i in range(mat_w)], up_border, down_border, color='black')

    def visualize(self, paths, class_):
        """
        see where the cell is
        """
        print("=" * 40)
        print("path length:", [len(path) - 1 for path in paths])
        print("=" * 40)
        plt.matshow(-self.map, cmap=plt.cm.hot)
        mat_w, mat_h = len(self.map[0]), len(self.map)
        self.plot_lines(mat_w, mat_h)

        # plot cluster
        axis_x, axis_y = [], []
        for i in sorted(list(self.empty_cells)):  # have deleted the obstacle cells
            x, y = self.decode(i)
            axis_x.append(x)
            axis_y.append(y)
        color_set = []
        for row in class_:
            # color_code = sum(np.where(row != 0)[0])
            classes = np.where(row != 0)[0]
            if len(classes) > 1:
                rgb_color = []
                for c in classes:
                    rgb_color.append(list(plt.cm.Set1(c)))
                color_set.append(np.mean(rgb_color, axis=0))
            else:
                color_code = classes[0]
                color_set.append(plt.cm.Set1(color_code))  # Set1有取值范围[0,7]

        plt.scatter(axis_y, axis_x, c=color_set, s=100, alpha=0.4)  # attention! x and y are reversed!

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
                x_2, y_2 = self.decode(path[j + 1])
                dx_ = x_2 - x_1
                dy_ = y_2 - y_1
                plt.arrow(y_1, x_1, dx=dy_, dy=dx_, width=0.01, ec=color, alpha=0.8,
                          fc=color,
                          head_width=0.2,
                          length_includes_head=True)  # 坐标系位置和矩阵cell位置表示是相反的
        plt.show()


def read_map(path):
    print("map: ", path)
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
    """
    DF: normal 2
    """
    # path = "../maps"
    # files = os.listdir(path)
    file_name = "../maps/4_11d.txt"
    map = read_map(file_name)
    results = []
    for combi in product(optional_params['f_weight'], optional_params['IW'], optional_params['WR'],
                         optional_params['n_agent'],
                         optional_params['heuristic']):
        my_params = {"f_weight": combi[0],
                     "f_option": "WA",
                     "DF_factor": 2,
                     "IW": combi[1],
                     "WR": combi[2],
                     "n_agent": combi[3],
                     "heuristic": combi[4],
                     "write_to_file": False,
                     "test_time": 1,
                     # "start": None,
                     "start": [(2, 1), (14, 11), (11, 0), (12, 4), (16, 0)],
                     "verbose": True,
                     "visualize": True}
        sol = MWRP(map, **my_params)
        optional = optional_params.keys()
        result_dict = sol.run(my_params['test_time'])
        results.append([my_params[o] for o in optional] + list(result_dict.values()))
    my_columns = ["f_weight", "IW", "WR", "n_agent", "heuristic", "total_time", "expands", "avg_max_paths",
                  "paths_len_std"]
    my_results = pd.DataFrame(results, columns=my_columns)
    my_results.to_excel("../results/experiment.xlsx")


main()
