"""
改进了LOS的判断逻辑
增加了LOS8选项
修复了make graph考虑情况不周全的问题，实现了LOS8 size11情况下的最优解
"""
import numpy as np
import time
import os
from copy import deepcopy
from collections import deque, defaultdict
from heapq import *
from Graph import Graph, Vertex
from MST_kruskal import MiniSpanTree_kruskal
import matplotlib.pyplot as plt


# 先假设是往上下左右看的, 然后移动cost就是单纯的距离，为1
# 之后考虑搞一个cost矩阵，size:n-1*n-1,代表在相邻cell之间移动的代价
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
    """

    def __init__(self, map_, start):
        self.map = map_
        self.h, self.w = len(self.map), len(self.map[0])
        self.start = self.encode(start[0], start[1])
        self.pq = PriorityQueue()
        self.empty_cells = set()
        self.LOS = {}
        self.APSP = {}
        self.APSP_d = {}
        self.edge_list = {}
        self.nodes = 0

    def run(self, test_times):
        # self.visualize([])
        self.initialize()
        seen = self.LOS[self.start]
        path = [self.start]
        print("LOS:", self.LOS)
        # print("APSP_d:", self.APSP_d)
        # print("APSP:", self.APSP[(43, 64)])
        start = time.perf_counter()
        for i in range(test_times):
            cur_state = State(path, seen)
            while not self.is_finish(cur_state):
                # self.visualize(cur_state.path)
                self.next_step(cur_state)
                cur_state = self.pq.pop_()
            self.visualize(cur_state.path)
            assert self.check_finish(cur_state.path), "路径有误！"
            seen = self.LOS[self.start]
            path = [self.start]
        print("running time:{} s, expanding nodes:{}".format(time.perf_counter() - start, self.nodes))

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
        _, near_watchers = self.make_graph(cur_state.seen, cur_state.path, next_step=True)
        for new_pos in near_watchers:
            new_x, new_y = self.decode(new_pos)
            a, b = cur_state.cur_pos, new_pos
            if a > b:
                a, b = b, a
            path = deepcopy(self.APSP[(a, b)])  # the path from a to b will go through
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
            # h_value = self.calc_MST_h(cur_seen, cur_path)
            h_value = 0
            # h_value = self.calc_agg_h(cur_seen, cur_path)
            A_star_v = h_value + len(cur_path)
            self.pq.push_(State(cur_path, cur_seen, A_star_v))
            self.nodes += 1

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
        graph, _ = self.make_graph(cur_seen, cur_path, next_step=False)
        choice, result = MiniSpanTree_kruskal(graph)
        return result

    def make_graph(self, cur_seen, cur_path, next_step):
        """
        generate disjoint graph from cur_state
        find next state which next to agent in disjoint graph
        next_step: when True, need to return near_watchers and need to considering white cell,
                   when False, return graph and ignore white and gray cell
        """
        agent_code = cur_path[-1]
        unseen = list(self.empty_cells - cur_seen)
        pivots, watcher = [], set()
        cell_group = {}  # the component where cell from
        g = Graph()
        near_watchers = set()  # the watcher close to agent, next step
        unseen.sort(key=lambda x: len(self.LOS[x]))  # |LOS4| ascending order
        # decide which cell to be pivot
        for p in unseen:
            if not (self.LOS[p] & watcher):  # if no replicate, new pivot
                pivots.append(p)
                temp = self.LOS[p]
                # temp.remove(p)  # 除去pivot本身
                watcher = watcher | temp  # here watcher including pivot itself
                for cell in temp:
                    cell_group[cell] = p
        cell_group[agent_code] = agent_code
        unreached = self.empty_cells - set(cell_group.keys())  # gray or white cell
        if next_step:  # need to consider white cell which means not in cur_seen
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
        for edge_num in edge_list.keys():
            link_edges = edge_list[edge_num]
            group_a = cell_group[edge_num]
            for link_edge in link_edges:
                group_b = cell_group[link_edge]
                if group_a == group_b:  # belong to the same component
                    g.addEdge(edge_num, link_edge, 0)
                    # print((edge_num, link_edge, 0))
                else:
                    a = edge_num
                    b = link_edge
                    if a > b:
                        a, b = b, a
                    g.addEdge(edge_num, link_edge, self.APSP_d[(a, b)])
                    # print(edge_num, link_edge, self.APSP_d[(a, b)])
                    if edge_num == agent_code:
                        near_watchers.add(link_edge)
        return g, near_watchers

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

    def initialize(self):
        """
        prepare two lookup tables for efficiency
        """
        for x in range(self.h):
            for y in range(self.w):
                if self.map[x, y] == 0:
                    code = self.encode(x, y)
                    self.empty_cells.add(code)
                    self.LOS[code] = self.LOS8(code)
                    for num in range(code + 1, self.h * self.w):  # 与其他所有点的最短距离
                        cell_x, cell_y = self.decode(num)
                        if self.map[cell_x, cell_y] == 1:
                            continue
                        path, d = self.minimum_d((x, y), (cell_x, cell_y))
                        a, b = code, num
                        # make sure a is smaller
                        assert a < b
                        self.APSP_d[(a, b)] = d
                        self.APSP[(a, b)] = path
                    # build edge list
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

    def visualize(self, path):
        """
        see where the cell is
        """
        print(f"length:{len(path)}", path)
        plt.matshow(-self.map, cmap=plt.cm.hot)
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
    # map = np.array([[1, 0, 0, 0],
    #                 [1, 0, 1, 1],
    #                 [0, 0, 0, 0],
    #                 [0, 1, 1, 0],
    #                 [0, 0, 0, 0]])
    # start = (2, 1)
    map = np.array([[0, 0, 0, 0, 0],
                    [1, 1, 0, 1, 1],
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 0, 0, 0]])
    start = (4, 0)
    # map = np.array([[1, 1, 0, 0, 0],
    #                 [1, 1, 0, 0, 1],
    #                 [1, 0, 0, 1, 1],
    #                 [0, 0, 1, 1, 1],
    #                 [0, 1, 1, 1, 1]])
    # start = (4, 0)
    path = "../../research/研究内容/maps"
    files = os.listdir(path)
    os.chdir(path)
    for file in files:
        print(file)
        start, map = read_map(file)
        test_times = 1
        sol = WatchmanRouteProblem(map, start)
        sol.run(test_times)
        break

main()
