from package import *


class State:
    def __init__(self, path, seen, A_star_v=0):
        self.path = path
        self.cur_pos = path[-1]
        self.seen = seen
        self.A_star_v = A_star_v

    def get_value(self):
        return self.A_star_v


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
    heuristic: select heuristic function
    """

    def __init__(self, map_, start, **params):
        self.map = map_
        self.h, self.w = len(self.map), len(self.map[0])
        self.start = self.encode(start[0], start[1])
        self.pq, self.empty_cells = PriorityQueue(), params.get("empty_cells")
        self.LOS, self.APSP, self.APSP_d = params.get("LOS"), params.get("APSP"), params.get("APSP_d")
        self.edge_list, self.nodes = params.get("edge_list"), 0
        self.f_weight, self.f_option = params.get("f_weight"), params.get("f_option")
        self.DF_factor = params.get("DF_factor")
        self.IW, self.WR = params.get("IW"), params.get("WR")
        self.heuristic = params.get("heuristic")
        self.silent = params.get("silent")

    def add_seen(self, seen, LOS):
        return seen | (LOS & self.empty_cells)

    def run(self):
        seen = self.add_seen(set(), self.LOS[self.start])
        start_path = [self.start]
        start = time.perf_counter()
        cur_state = State(start_path, seen)
        while not self.is_finish(cur_state):
            self.next_step(cur_state)
            cur_state = self.pq.pop_()
        if not self.silent:
            print(
                "running time:{} s, expanding nodes:{}, start point:{}, path length: {}".format(
                    time.perf_counter() - start,
                    self.nodes,
                    self.decode(self.start),
                    len(cur_state.path) - 1))
        return cur_state.path

    def next_step(self, cur_state):
        _, near_watchers = self.make_graph(cur_state.seen, cur_state.path, IW=self.IW, WR=self.WR,
                                           BJP_DF=self.DF_factor)
        for new_pos in near_watchers:
            new_x, new_y = self.decode(new_pos)
            path = self.get_APSP(cur_state.cur_pos, new_pos, distance=False).copy()  # the path will go through
            # print(f"cur_pos:{a} to new_pos:{b}: {path}")
            assert 0 <= new_x < self.h and 0 <= new_y < self.w and self.map[new_x][new_y] != 1
            cur_path = cur_state.path.copy()
            if path[-1] != new_pos:
                path = path[::-1]
            path.remove(cur_state.cur_pos)
            cur_path.extend(path)
            cur_seen = cur_state.seen.copy()
            for cell in path:
                cur_seen = self.add_seen(cur_seen, self.LOS[cell])
            if self.heuristic == "MST":
                h_value = self.calc_MST_h(cur_seen, cur_path)
            elif self.heuristic == "TSP":
                h_value = self.calc_TSP_h(cur_seen, cur_path)
            elif self.heuristic == "agg_h":
                h_value = self.calc_agg_h(cur_seen, cur_path)
            else:
                h_value = 0
            A_star_v = self.calc_A_stat_v(len(cur_path), h_value, w=self.f_weight, option=self.f_option)
            self.pq.push_(State(cur_path, cur_seen, A_star_v))
            self.nodes += 1

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
        distance_matrix, _ = self.make_graph(cur_seen, cur_path, IW=self.IW, WR=self.WR, BJP_DF=self.DF_factor)
        # choice, result = MiniSpanTree_kruskal(graph)
        X = csr_matrix(distance_matrix)
        Tcsr = minimum_spanning_tree(X)
        return Tcsr.toarray().astype(int).sum()

    def calc_TSP_h(self, cur_seen, cur_path):
        distance_matrix, _ = self.make_graph(cur_seen, cur_path, IW=self.IW, WR=self.WR, TSP=True,
                                             BJP_DF=self.DF_factor)
        permutation, distance = solve_tsp_local_search(distance_matrix)
        return int(distance % 1e5)

    def make_graph(self, cur_seen, cur_path, IW=True, WR=True, TSP=False, BJP_DF=0):
        """
        generate disjoint graph from cur_state
        find next state which next to agent in disjoint graph
        next_step: when True, need to return near_watchers and need to considering white cell,
                   when False, return graph and ignore white and gray cell
        IW: if True white cells will be ignored
        BJP_DF:
            when True Bounding the Jump Points, near_watcher whose w > DF*epsilon_s will be pruned
            epsilon_s: the cost of the edge of the closest jump point from S.location
        WR: Weakly Redundant pivots will be moved, reduce the search size of tree
        TSP: return matrix for TSP heuristic
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
        if WR:  # if one path include some other pivot's watcher, delete that pivot
            deleted = []
            for p1, p_path in pivots_path.items():
                for p2 in pivots_path:
                    if p1 == p2 or p1 in deleted or p2 in deleted:
                        continue
                    p2_watcher = self.LOS[p2]
                    if set(p_path) & p2_watcher:
                        deleted.append(p2)
                        pivots.remove(p2)
                        watcher = watcher - p2_watcher
                        for cell in p2_watcher:
                            cell_group.pop(cell)
        cell_group[agent_code] = agent_code
        unreached = self.empty_cells - set(cell_group.keys())  # gray or white cell
        if not IW:  # need to consider white cell which means not in cur_seen
            white_cells = set()
            for cell in unreached:
                if cell not in cur_seen:
                    white_cells.add(cell)
            unreached = unreached - white_cells  # leave only gray cells in it
            while white_cells:
                p = white_cells.pop()
                pivots.append(p)
                temp = self.LOS[p] & (white_cells | {p})
                watcher = watcher | temp
                for cell in temp:
                    cell_group[cell] = p
                white_cells = white_cells - temp
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
            pivots_map = [edge_map[i] for i in pivots]  # 各个pivot对应的map
            # 构建只有pivots节点的距离矩阵
            matrix_for_pivot = APSP_m[pivots_map][:, pivots_map]  # 取出pivots对应的行和列, pivots第一个元素是agent位置
            # 所以该矩阵的第一行一定是agent
            m_len = len(matrix_for_pivot)
            tsp_m = np.zeros((m_len + 1, m_len + 1))
            tsp_m[:m_len, :m_len] = matrix_for_pivot
            tsp_m[m_len, :] = tsp_m[:, m_len] = [1e-2] + [1e5] * m_len
            return tsp_m, near_watchers
        else:
            return distance_matrix, near_watchers

    def floyd_APSP(self, distance_matrix):
        m = distance_matrix.copy()
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
        # edge_list = deepcopy(self.edge_list)
        edge_list = {}
        for key, value in self.edge_list.items():
            edge_list[key] = value.copy()
        for code in need_to_compact:
            temp = edge_list.pop(code)  # delete cell from edge_list
            for edge_num in temp:
                to_be_add = [i for i in temp if i != edge_num]
                edge_list[edge_num].remove(code)  # delete cell
                edge_list[edge_num].extend(to_be_add)
                edge_list[edge_num] = list(set(edge_list[edge_num]))  # prevent duplicates
        return edge_list

    def is_finish(self, cur_state):
        """
        all empty cells have been added in seen
        :return:
        """
        return len(cur_state.seen) == len(self.empty_cells)
