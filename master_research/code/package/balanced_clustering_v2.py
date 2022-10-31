"""
adapted to multi-class
"""
import numpy as np
import math
from collections import Counter
from copy import deepcopy


class Cluster:
    def __init__(self, central_idx):
        self.central_idx = central_idx
        self.reachable = set()
        self.members = [central_idx]
        self.predecessor = {}
        # self.extensible = []

    # def add_extensible(self, extensible):
    #     self.extensible.append(extensible)
    #     # 如何判断是不是extensible/边界点, 当与障碍物相接时不算，先不考虑

    def add_member(self, cell, reachable):
        """
        add new member to cluster and remove it from reachable
        update reachable
        """
        self.members.append(cell)
        self.remove_reachable([cell], extensible=True)
        self.add_reachable(cell, reachable)

    def add_reachable(self, predecessor, cells):
        """
        add new reachable cells, and they are all connecting to predecessor
        """
        cells = set(cells) - set(self.members)  # exclude existing cells in cluster
        self.reachable |= cells
        for cell in cells:
            self.predecessor[cell] = predecessor

    def get_predecessor(self, cell):
        return self.predecessor[cell]

    def remove_reachable(self, cells, extensible=False):
        """
        case1: this cell has been owned by other cluster, need to remove
        case2: this cell transform from reachable to extensible
        if case1: this cell is not predecessor in this cluster anymore. need to remove from "self.predecessor"
        """
        for cell in cells:
            if not extensible:  # If extensible is True, no need to pop
                self.predecessor.pop(cell)
            assert cell in self.reachable, "not exist"
            self.reachable -= {cell}


class MyKmeans2:
    """
    可以基于这个去做一些调整
    """
    def __init__(self, k, non_zero_idx, APSP_d, edge_list, n_start_pos, max_iter=300):
        """
        :param k:
        :param non_zero_idx: get the correct cell number after removing obstacles
                            if non_zero_idx[0]=5, means the first non-obstacle cell is No.5 cell
        :param edge_list:  check whether this cell pair is adjacent
        :param max_iter:
        """
        self.k = k
        self.n_start_pos = n_start_pos
        self.max_iter = max_iter
        self.features_count = -1
        self.clusters = []
        self.n_sample = None
        self.balanced_size = None
        self.non_zero_idx = non_zero_idx
        self.edge_list = edge_list
        self.APSP_d = APSP_d
        self.counter = None

    def initialize_centroids(self, distances) -> None:
        """
        :param distances: the internal modification will be synchronized to the external, no need to return
        :type distances: dict
        """
        # 初始化聚类中心
        for i in range(self.k):
            # idx = self.non_zero_idx[self.n_sample//self.k*i]  # 等距离选择聚类中心
            idx = self.n_start_pos[i]  # 选择agent的出发点作为各个聚类的中心
            self.clusters.append(Cluster(idx))
            distances[idx] = 0
            self.clusters[i].add_reachable(idx, self.edge_list[idx])

    def fit_predict(self, data):
        """
        :param data: numpy数组，约定shape为：(数据数量，数据维度), 已排除障碍物cell
        :type data: np.ndarray
        every cluster will have a responding spanning tree and every node in C will have a predecessor
        distances: distances[p]表示p点与所属根节点r之间的总路径长度，当不属于任何簇时，为-1
                   各簇根节点距离为0
        """
        self.n_sample = data.shape[0]
        self.features_count = data.shape[1]
        self.balanced_size = np.ceil(self.n_sample / self.k)
        distances = {idx: -1 for idx in self.non_zero_idx}
        mapping = {cell_num: idx for idx, cell_num in enumerate(self.non_zero_idx)}
        # mapping[5]=2 means the No.5 cell is the 2nd non_zero cell
        self.initialize_centroids(distances)

        for i in range(1, self.max_iter+1):
            # 清空聚类
            is_change = True
            while -1 in distances.values():
                assert is_change is True, "failed by clustering algorithm"
                # if not is_change:
                #     print("clustering failed, restart")
                is_change = False
                for cluster in self.clusters:  # ===========================
                    candidates = cluster.reachable.copy()  # reachable
                    for c in candidates:
                        pred = cluster.get_predecessor(c)
                        temp_d = distances[pred] + 1  # the distance between current cell and root
                        if distances[c] != -1 and temp_d > distances[c]:
                            # 已经被归属其他簇，且距离小于当前簇，则不满足多分类条件
                            cluster.remove_reachable([c])
                            continue
                        cluster.add_member(c, self.edge_list[c])
                        is_change = True
                        distances[c] = temp_d
            # 记录前一次聚类完成后的中心
            prev_centroids = [cluster.central_idx for cluster in self.clusters]
            # 更新中心, distances也需要重置
            new_clusters = []
            new_distances = {idx: -1 for idx in self.non_zero_idx}
            for num, cluster in enumerate(self.clusters):
                member_idx = [mapping[c] for c in cluster.members]
                cluster_data = data[member_idx]
                centroid = np.average(cluster_data, axis=0)
                new_cen_idx = self.non_zero_idx[np.linalg.norm(data - centroid, axis=1).argmin()]
                new_clusters.append(Cluster(new_cen_idx))
                new_distances[new_cen_idx] = 0
                new_clusters[num].add_reachable(new_cen_idx, self.edge_list[new_cen_idx])
            # 检测两次聚类中心的变化
            cur_centroids = [cluster.central_idx for cluster in new_clusters]
            if prev_centroids == cur_centroids or i == self.max_iter:
                # the 2-nd condition is to solve the situation of over max-iteration
                print(f"after {i} iterations, centroids no longer change")
                result = self.get_clustering_result(mapping)
                # predecessors中有一部分是预备加入的点而不是已经加入簇的，需要排除
                for cluster in self.clusters:
                    cluster.predecessor = {key: value for key, value in cluster.predecessor.items()
                                           if key in cluster.members}
                return result
                # return self.balanced_connect_processing(result)
            self.clusters = new_clusters
            distances = new_distances
        # else:
        #     raise RuntimeError("max_iteration over, clustering failed!!")

    def get_clustering_result(self, mapping):
        clustering = np.zeros((self.n_sample, self.k))
        for cluster_code, cluster in enumerate(self.clusters):
            idxes = [mapping[c] for c in cluster.members]
            clustering[idxes, cluster_code] = 1
        return clustering

    def balanced_connect_processing(self, clustering):
        """
        如果2个点属于不同的类别，且有edge连着，判断是否进行move
        balanced value of a move:
        假设a属于Cr，b属于Cs依次判断以下条件
        value=2：如果|Cr|>balance_size and |Cs|<balance_size
        value=1: if |Cr| > |Cs|+1
        value=0 if |Cr|=|Cs|+1
        value=-1 if |Cr|<=|Cs|

        WM(C)=||C|-balance_size|
        weight value of move=WM|Cr|+WM|Cs|-WM|Cr-pi|-WM|Cs+pi|  (pi,当前点)
        ========
        执行move的条件：balanced value>0 or (balanced value=0 and weight value of move > 0)
        do move operation until no more remain
        """
        tolerance_size = 3
        self.counter = Counter(clustering)
        mapping = {cell_num: idx for idx, cell_num in enumerate(self.non_zero_idx)}
        # mapping[5]=2 means the No.5 cell is the 2nd non_zero cell
        for i in range(self.max_iter):
            for real_cell_idx, edges in self.edge_list.items():  # plan to assign key-th cell to other cluster
                idx1 = mapping[real_cell_idx]
                cls_1 = clustering[idx1]
                candidate = edges
                for nxt_cell in candidate:
                    idx2 = mapping[nxt_cell]
                    cls_2 = clustering[idx2]
                    if cls_1 == cls_2:  # real_cell_idx已经被分配过了
                        continue
                    # from different cluster, do assign operation depending on feasibility checking
                    cls_1_size, cls_2_size = self.counter[cls_1], self.counter[cls_2]
                    if cls_1_size > self.balanced_size > cls_2_size:
                        if self.feasibility_checking(real_cell_idx, cls_1):
                            self.assign_to_cluster(clustering, idx1, cls_1, cls_2)
                            break   # 如果分配成功，cls_1此时已经改变，需要从外层循环开始重新加载
                        # else:
                        #     self.assign_to_cluster(clustering, idx1, cls_1, cls_2, common=True)
                    elif cls_1_size - cls_2_size > tolerance_size:
                        if self.feasibility_checking(real_cell_idx, cls_1):
                            self.assign_to_cluster(clustering, idx1, cls_1, cls_2)
                            break
                    elif 0 <= cls_1_size - cls_2_size <= tolerance_size:
                        weight_value = abs(cls_1_size - self.balanced_size) + abs(cls_2_size - self.balanced_size) \
                                       - abs(cls_1_size - 1 - self.balanced_size) - abs(
                            cls_2_size + 1 - self.balanced_size)
                        if weight_value > 0:
                            if self.feasibility_checking(real_cell_idx, cls_1):
                                self.assign_to_cluster(clustering, idx1, cls_1, cls_2)
                                break
                    elif cls_1_size - cls_2_size < 0:
                        continue
        return clustering

    def feasibility_checking(self, real_cell_idx, cls_1):
        """
        to check if cls_1 lose idx, are other cells influenced by it
        if no successors, assign to other cluster is no effect
        if yes, be sure successors can find any other predecessor in cls_1
        """
        # print(f"checking {idx}")
        cur_cluster = deepcopy(self.clusters[cls_1])
        successors = [success for success, pred in cur_cluster.predecessor.items() if pred == real_cell_idx]
        if successors:
            members = cur_cluster.members
            for suc in successors:
                for new_pred in members:
                    if new_pred == real_cell_idx or suc == new_pred:
                        continue
                    if self.APSP_d[suc, new_pred] == 1 and cur_cluster.predecessor[new_pred] != suc:
                        # 相邻的其他节点选定为新的前驱节点,但不能是该节点的后继节点，会造成互相连接
                        cur_cluster.predecessor[suc] = new_pred
                        break
                else:
                    # print(f"succ {suc} can't find new predecessor")
                    return False
        return True

    def assign_to_cluster(self, clustering, idx, cls_1, cls_2, common=False):
        """
        assign idx_cell from cls_1 to cls_2
        """
        real_cell_idx = self.non_zero_idx[idx]
        if not common:
            # remove cell from cls_1
            self.clusters[cls_1].members.remove(real_cell_idx)  # 出错地
            self.clusters[cls_1].predecessor.pop(real_cell_idx)
            # add to cls_2
            clustering[idx] = cls_2
            cur_members = self.clusters[cls_2].members
            pred_idx = np.abs(np.array(cur_members) - real_cell_idx).argmin()
            self.clusters[cls_2].predecessor[real_cell_idx] = cur_members[pred_idx]
            self.clusters[cls_2].members.append(real_cell_idx)
            self.counter[cls_1] -= 1
            self.counter[cls_2] += 1
            # print(f"assign {(real_cell_idx // 4, real_cell_idx % 4)} from cls_{cls_1} to cls_{cls_2}")
        # else:
        #     # add to cls_2
        #     clustering[idx] = self.common
        #     cur_members = self.clusters[cls_2].members
        #     pred_idx = np.abs(np.array(cur_members) - real_cell_idx).argmin()
        #     self.clusters[cls_2].predecessor[real_cell_idx] = cur_members[pred_idx]
        #     self.clusters[cls_2].members.append(real_cell_idx)
        #     self.counter[cls_2] += 1
        #     print(f"assign {(real_cell_idx // 4, real_cell_idx % 4)} from cls_{cls_1} to cls_{cls_2} with common")
