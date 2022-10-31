"""
2022.9.19
add random start
revise bug of multiple test time(doesn't run correctly)
"""
import numpy as np
import time
import math
import os
from copy import deepcopy
from collections import deque, defaultdict
from heapq import *
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, floyd_warshall
import matplotlib.pyplot as plt
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_local_search
from tqdm import trange


class PriorityQueue:
    def __init__(self):
        self.pq = defaultdict(list)
        self.min_pos = []

    def push_(self, state):
        self.pq[state.A_star_v].append(state)
        heappush(self.min_pos, state.A_star_v)

    def pop_(self):
        return self.pq[heappop(self.min_pos)].pop()

    def is_empty(self):
        return True if len(self.min_pos) == 0 else False


class State:
    def __init__(self, path, seen, A_star_v=0):
        self.path = path
        self.cur_pos = path[-1]
        self.seen = seen
        self.A_star_v = A_star_v


class WatchmanRouteProblem:
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
        self.map = map_
        self.h, self.w = len(self.map), len(self.map[0])
        self.start = self.encode(start[0], start[1])
        self.pq, self.empty_cells = PriorityQueue(), set()
        self.LOS, self.APSP, self.APSP_d = {}, {}, {}
        self.edge_list, self.nodes = {}, 0
        self.f_weight, self.f_option = params.get("f_weight"), params.get("f_option")
        self.DF_factor = params.get("DF_factor")
        self.IW, self.WP = params.get("IW"), params.get("WP")
        self.heuristic = params.get("heuristic")
        self.obstacles = 0

    def run(self, test_times):
        # self.visualize([])
        np.random.seed(42)
        self.initialize()
        seen = self.LOS[self.start]
        path = [self.start]
        # print("LOS:", self.LOS)
        start = time.perf_counter()
        paths_len = []
        for i in trange(test_times):
            cur_state = State(path, seen)
            while not self.is_finish(cur_state):
                # self.visualize(cur_state.path)
                self.next_step(cur_state)
                cur_state = self.pq.pop_()
            self.visualize(cur_state.path)
            paths_len.append(len(cur_state.path)-1)
            assert self.check_finish(cur_state.path), "路径有误！"
            self.start = np.random.choice(list(self.empty_cells), 1, replace=False)[0]
            seen = self.LOS[self.start]
            path = [self.start]
            self.pq = PriorityQueue()
        print(f"map_size:{(self.h, self.w)}, obstacles:{self.obstacles}")
        print("running time:{:.3f} s, expanding nodes:{}, avg_path_len:{}".
              format((time.perf_counter() - start), self.nodes, np.mean(paths_len)))

    def check_finish(self, path):
        map_ = self.map.copy()
        axis_x = []
        axis_y = []
        for pos in path:
            sights = self.LOS[pos]
            for sight in sights:
                x, y = self.decode(sight)
                axis_x.append(x)
                axis_y.append(y)
        map_[axis_x, axis_y] = 1
        return map_.sum() == self.w*self.h

    def next_step(self, cur_state):
        _, near_watchers = self.make_graph(cur_state.seen, cur_state.path, IW=self.IW, WP=self.WP, BJP_DF=self.DF_factor)
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
            if self.heuristic == "MST":
                h_value = self.calc_MST_h(cur_seen, cur_path)
            elif self.heuristic == "TSP":
                h_value = self.calc_TSP_h(cur_seen, cur_path)
            elif self.heuristic == "agg_h":
                h_value = self.calc_agg_h(cur_seen, cur_path)
            else:
                h_value = 0

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

    def LOS4(self, location):
        """
        all cells can be seen from location including itself
        """
        x, y = self.decode(location)
        can_see = set()
        can_see.add(location)
        assert self.map[x, y] == 0, "illegal position!"
        h, w = len(self.map), len(self.map[0])
        left = right = up = down = True
        times = 1
        while any([left, right, up, down]):
            left_border, right_border = y - times, y + times
            up_border, down_border = x - times, x + times
            if left and left_border >= 0 and self.map[x][left_border] == 0:
                can_see.add(self.encode(x, left_border))
            else:
                left = False
            if right and right_border < w and self.map[x][right_border] == 0:
                can_see.add(self.encode(x, right_border))
            else:
                right = False
            if up and up_border >= 0 and self.map[up_border][y] == 0:
                can_see.add(self.encode(up_border, y))
            else:
                up = False
            if down and down_border < h and self.map[down_border][y] == 0:
                can_see.add(self.encode(down_border, y))
            else:
                down = False
            times += 1
        return can_see

    def LOS8(self, location):
        """
        including diagonal line compared with LOS4
        """
        x, y = self.decode(location)
        can_see = set()
        can_see.add(location)
        assert self.map[x, y] == 0, "illegal position!"
        h, w = len(self.map), len(self.map[0])
        left = right = up = down = True
        upleft = downleft = upright = downright = True
        times = 1
        while any([left, right, up, down, upleft, downleft, upright, downright]):
            left_border, right_border = y - times, y + times
            up_border, down_border = x - times, x + times
            if left and left_border >= 0 and self.map[x][left_border] == 0:
                can_see.add(self.encode(x, left_border))
            else:
                left = False
            if right and right_border < w and self.map[x][right_border] == 0:
                can_see.add(self.encode(x, right_border))
            else:
                right = False
            if up and up_border >= 0 and self.map[up_border][y] == 0:
                can_see.add(self.encode(up_border, y))
            else:
                up = False
            if down and down_border < h and self.map[down_border][y] == 0:
                can_see.add(self.encode(down_border, y))
            else:
                down = False
            if upleft and left_border >= 0 and up_border >= 0 and \
                    self.map[up_border][left_border] == 0:
                can_see.add(self.encode(up_border, left_border))
            else:
                upleft = False
            if downleft and left_border >= 0 and down_border < h and \
                    self.map[down_border][left_border] == 0:
                can_see.add(self.encode(down_border, left_border))
            else:
                downleft = False
            if upright and right_border < w and up_border >= 0 and \
                    self.map[up_border][right_border] == 0:
                can_see.add(self.encode(up_border, right_border))
            else:
                upright = False
            if downright and right_border < w and down_border < h and \
                    self.map[down_border][right_border] == 0:
                can_see.add(self.encode(down_border, right_border))
            else:
                downright = False
            times += 1
        return can_see

    def calc_agg_h(self, seen, path):
        unseen = self.empty_cells - seen
        cur_location = path[-1]
        agg_h = float('-inf')
        for p in unseen:
            single_h = self.singleton_h(cur_location, p)
            agg_h = max(agg_h, single_h)
        return agg_h

    def singleton_h(self, cur_location, p):
        """
        cur_location: where agent is
        :param p: one cell from unseen
        """
        cells = self.LOS[p]
        min_h = float('inf')
        for cell in cells:
            a, b = cur_location, cell
            if a > b:
                a, b = b, a
            min_h = min(min_h, self.APSP_d.get((a, b), float('inf')))
        return min_h

    def calc_MST_h(self, cur_seen, cur_path):
        distance_matrix, _ = self.make_graph(cur_seen, cur_path, IW=self.IW, WP=self.WP, BJP_DF=self.DF_factor)
        # choice, result = MiniSpanTree_kruskal(graph)
        X = csr_matrix(distance_matrix)
        Tcsr = minimum_spanning_tree(X)
        return Tcsr.toarray().astype(int).sum()

    def calc_TSP_h(self, cur_seen, cur_path):
        distance_matrix, _ = self.make_graph(cur_seen, cur_path, IW=self.IW, WP=self.WP, TSP=True, BJP_DF=self.DF_factor)
        permutation, distance = solve_tsp_local_search(distance_matrix)
        return int(distance % 1e5)

    def make_graph(self, cur_seen, cur_path, IW=True, WP=True, TSP=False, BJP_DF=0):
        """
        generate disjoint graph from cur_state
        find next state which next to agent in disjoint graph
        next_step: when True, need to return near_watchers and need to considering white cell,
                   when False, return graph and ignore white and gray cell
        IW: if True white cells will be ignored
        BJP_DF:
            when True Bounding the Jump Points, near_watcher whose w > DF*epsilon_s will be pruned
            epsilon_s: the cost of the edge of the closest jump point from S.location
        """
        agent_code = cur_path[-1]
        unseen = list(self.empty_cells - cur_seen)
        unseen.sort(key=lambda x: len(self.LOS[x]))  # |LOS4| ascending order
        pivots, watcher = [agent_code], set()
        cell_group = {}  # the component where cell from
        near_watchers, near_watchers_APSP_d = [], []  # the watcher close to agent, next step
        # decide which cell to be pivot
        pivots_path = {}  # the path from pos of agent to pivot
        for p in unseen:
            if not (self.LOS[p] & watcher):  # if no replicate, new pivot
                pivots.append(p)
                temp = self.LOS[p]
                # temp.remove(p)  # 除去pivot本身
                watcher = watcher | temp  # here watcher including pivot itself
                for cell in temp:
                    cell_group[cell] = p
                pivots_path[p] = self.get_APSP(agent_code, p, distance=False)
        if WP:  # if one path include some other pivot's watcher, delete that pivot
            deleted = []
            for p1, p_path in pivots_path.items():
                for p2 in pivots_path:
                    if p1 == p2 or p1 in deleted or p2 in deleted:
                        continue
                    p2_watcher = self.LOS[p2]
                    if set(p_path) & p2_watcher:
                        deleted.append(p2)
                        pivots.remove(p2)
                        watcher = watcher-p2_watcher
                        for cell in p2_watcher:
                            cell_group.pop(cell)
        cell_group[agent_code] = agent_code
        unreached = self.empty_cells - set(cell_group.keys())  # gray or white cell
        if not IW:  # need to consider white cell which means not in cur_seen
            white_cells = set()
            for cell in unreached:
                if cell not in cur_seen:
                    white_cells.add(cell)
            unreached = unreached-white_cells  # leave only gray cells in it
            while white_cells:
                p = white_cells.pop()
                pivots.append(p)
                temp = self.LOS[p] & (white_cells | {p})
                watcher = watcher | temp
                for cell in temp:
                    cell_group[cell] = p
                white_cells = white_cells-temp
        edge_list = self.compact_edge(unreached)
        edge_map = {edge: num for num, edge in enumerate(edge_list)}  # simplify the matrix size
        distance_matrix = np.zeros((len(edge_map), len(edge_map)))
        for edge_num in edge_list.keys():
            link_edges = edge_list[edge_num]
            group_a = cell_group[edge_num]
            map_edge_num = edge_map[edge_num]
            for link_edge in link_edges:
                group_b = cell_group[link_edge]
                map_link_edge = edge_map[link_edge]
                if group_a == group_b:  # belong to the same component
                    distance_matrix[map_edge_num, map_link_edge] = 1e-2
                else:
                    APSP_d = self.get_APSP(edge_num, link_edge, distance=True)
                    distance_matrix[map_edge_num, map_link_edge] = APSP_d
                    if edge_num == agent_code:
                        near_watchers.append(link_edge)
                        near_watchers_APSP_d.append(APSP_d)
        if BJP_DF and near_watchers_APSP_d:  # not NULL
            epsilon_s = min(near_watchers_APSP_d)
            temp = []
            for i in range(len(near_watchers_APSP_d)):
                if near_watchers_APSP_d[i] <= BJP_DF * epsilon_s:
                    temp.append(near_watchers[i])
            near_watchers = temp
        if TSP:
            APSP_m = self.floyd_APSP(distance_matrix)
            pivots_map = [edge_map[i] for i in pivots]   # 各个pivot对应的map
            # 构建只有pivots节点的距离矩阵
            matrix_for_pivot = APSP_m[pivots_map][:, pivots_map]  # 取出pivots对应的行和列, pivots第一个元素是agent位置
            # 所以该矩阵的第一行一定是agent
            m_len = len(matrix_for_pivot)
            tsp_m = np.zeros((m_len+1, m_len+1))
            tsp_m[:m_len, :m_len] = matrix_for_pivot
            tsp_m[m_len, :] = tsp_m[:, m_len] = [1e-2]+[1e5]*m_len
            return tsp_m, near_watchers
        else:
            return distance_matrix, near_watchers

    def floyd_APSP(self, distance_matrix):
        m = deepcopy(distance_matrix)
        length = len(m)
        for k in range(length):
            for v in range(length):
                for w in range(length):
                    if v != w and m[v][k] > 0 and m[k][w] > 0 and (m[v][w] > m[v][k] + m[k][w] or m[v][w] == 0):
                        m[v][w] = m[v][k] + m[k][w]
        return m

    def get_APSP(self, a, b, distance=True):
        if a > b:
            a, b = b, a
        return self.APSP_d[(a, b)] if distance else self.APSP[(a, b)]

    def compact_edge(self, need_to_compact):
        """
        compact cells in need_to_compact
        """
        edge_list = deepcopy(self.edge_list)
        for code in need_to_compact:
            temp = edge_list.pop(code)  # delete cell from edge_list
            for edge_num in temp:
                to_be_add = [i for i in temp if i != edge_num]
                edge_list[edge_num].remove(code)  # delete cell
                edge_list[edge_num].extend(to_be_add)
                edge_list[edge_num] = list(set(edge_list[edge_num]))  # prevent duplicates
        return edge_list

    def get_path_from_pred(self, pred_m, start, end):
        path = [end]
        pred = pred_m[start, end]
        while pred != start:
            path.append(pred)
            pred = pred_m[start, pred]
        path.append(start)
        return path[::-1]

    def initialize(self):
        """
        prepare two lookup tables for efficiency
        """
        adjacent_matrix = np.zeros((self.h * self.w, self.h * self.w))
        for x in range(self.h):
            for y in range(self.w):
                if self.map[x, y] == 0:
                    code = self.encode(x, y)
                    self.empty_cells.add(code)
                    self.LOS[code] = self.LOS4(code)
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
                else:
                    self.obstacles += 1
        # build APSP and APSP_d
        graph = csr_matrix(adjacent_matrix)
        dist_matrix, predecessors = floyd_warshall(csgraph=graph, directed=False, return_predecessors=True)
        temp_list = list(self.empty_cells)
        temp_list.sort()
        for i in range(len(temp_list)):
            a = temp_list[i]
            for j in range(i + 1, len(temp_list)):
                b = temp_list[j]
                self.APSP[(a, b)] = self.get_path_from_pred(predecessors, a, b)
                self.APSP_d[(a, b)] = int(dist_matrix[a, b])
        del temp_list, dist_matrix, predecessors

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
            if (x, y) == (target_x, target_y):
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

    def is_finish(self, cur_state):
        """
        all empty cells have been added in seen
        :return:
        """
        return len(cur_state.seen) == len(self.empty_cells)

    def plot_lines(self, mat_w, mat_h):
        left_border = up_border = -0.5
        right_border, down_border = mat_w - 0.5, mat_h - 0.5
        plt.hlines([i - 0.5 for i in range(mat_h)], left_border, right_border, color='black')
        plt.vlines([i - 0.5 for i in range(mat_w)], up_border, down_border, color='black')

    def visualize(self, path):
        """
        see where the cell is
        """
        print(f"length:{len(path)-1}", path)
        plt.matshow(-self.map, cmap=plt.cm.hot)
        mat_w, mat_h = len(self.map[0]), len(self.map)
        self.plot_lines(mat_w, mat_h)
        start_x, start_y = self.decode(self.start)
        plt.text(start_y, start_x, s='start', fontsize='large', ha='center', va='center')
        length = len(path)
        for i in range(length - 1):
            x_1, y_1 = self.decode(path[i])
            x_2, y_2 = self.decode(path[i+1])
            dx_ = x_2 - x_1
            dy_ = y_2 - y_1
            plt.arrow(y_1, x_1, dx=dy_, dy=dx_, width=0.01, ec='red', alpha=1,
                      fc='red',
                      head_width=0.2,
                      length_includes_head=True)  # 坐标系位置和矩阵cell位置表示是相反的
        plt.show()


def read_map(path):
    matrix = []
    start_point = None
    with open(path) as f:
        for row, line in enumerate(f.readlines()[4:]):  # 前3行无用
            line = line.strip('\n')
            temp = []
            for col, alpha in enumerate(line):
                if alpha == '.':
                    temp.append(0)
                    if not start_point:
                        start_point = (row, col)
                else:
                    temp.append(1)
            matrix.append(temp)
    return start_point, np.array(matrix)


def main():
    map = np.array([[1, 0, 0, 0],
                    [1, 0, 1, 1],
                    [0, 0, 0, 0],
                    [0, 1, 1, 0],
                    [0, 0, 0, 0]])
    start = (2, 1)
    # map = np.array([[0, 0, 0, 0, 0],
    #                 [1, 1, 0, 1, 1],
    #                 [0, 0, 0, 0, 0],
    #                 [0, 1, 1, 1, 0],
    #                 [0, 1, 0, 0, 0]])
    # start = (4, 0)
    # map = np.array([[1, 1, 0, 0, 0],
    #                 [1, 1, 0, 0, 1],
    #                 [1, 0, 0, 1, 1],
    #                 [0, 0, 1, 1, 1],
    #                 [0, 1, 1, 1, 1]])
    # start = (4, 0)
    path = r"../maps"
    files = os.listdir(path)
    os.chdir(path)
    params = {"f_weight": 1, "f_option": "WA",
              "DF_factor": 2, "IW": True, "WP": True,
              "heuristic": "agg_h"}
    for file in files:
        print(file)
        if file != "0_lak101d.map":
            continue
        start, map = read_map(file)
        test_times = 10
        sol = WatchmanRouteProblem(map, start, **params)
        sol.run(test_times)
        break

main()
