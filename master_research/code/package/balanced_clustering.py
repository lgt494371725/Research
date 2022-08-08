from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
import math
from collections import Counter
from copy import deepcopy


# https://www.scipopt.org/


def get_even_clusters(X, n_clusters):
    """
    result seems bad
    会出现不联通问题,还有,单纯选择距离最远的cell进行再分配不行,应该要再考虑到这些cell离新簇的距离要尽可能接近
    """
    cluster_size = int(np.ceil(len(X) / n_clusters))
    kmeans = KMeans(n_clusters)
    clusters = kmeans.fit_predict(X)
    # centers = kmeans.cluster_centers_
    # centers = centers.reshape(-1, 1, X.shape[-1]).repeat(cluster_size, 1).reshape(-1, X.shape[-1])
    # distance_matrix = cdist(X, centers)
    # clusters = linear_sum_assignment(distance_matrix)[1]//cluster_size

    # 根据每个簇的size进行排序，从大的先开始重新分配
    # 计算簇内的cell与簇中心的距离，从小到大排序，超过指定cluster_size的cell则分配给其他cluster
    # 分配给除自身外的最近cluster，size最小的cluster不执行分配操作
    centers = kmeans.cluster_centers_
    clusters_size = []
    clusters_idx = []
    for num in range(n_clusters):
        cur_cluster_idx = np.where(clusters == num)[0]
        clusters_size.append(len(cur_cluster_idx))
        clusters_idx.append(cur_cluster_idx)
    size_rank = np.argsort(clusters_size)[::-1]
    for r in size_rank[:-1]:  # skip smallest cluster
        if clusters_size[r] < cluster_size:
            continue
        temp_center = centers[[r]]
        temp_cluster_idx = clusters_idx[r]
        temp_cells = X[temp_cluster_idx, :]
        # if r==1 and n_cluster=4,then other clusters are 0,2,3
        # other_center[0] means 0-th cluster, [1] means 2 and [2] means 3
        cluster_mapping = [i for i in range(n_clusters) if i != r]
        other_centers = centers[cluster_mapping]
        distances = cdist(temp_cells, temp_center).squeeze()  # cdist return 2-d array
        low_rank_cell_idx = np.argsort(distances)[cluster_size:]
        reassign_cell_idx = np.array([temp_cluster_idx[c] for c in low_rank_cell_idx])
        distances_to_new_c = cdist(temp_cells[low_rank_cell_idx, :], other_centers)
        new_cluster = np.argsort(distances_to_new_c)[:, 0]
        new_cluster = np.array([cluster_mapping[c] for c in new_cluster])
        assert len(new_cluster) == len(reassign_cell_idx)
        clusters[reassign_cell_idx] = new_cluster  # 待定
    return clusters


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
        self.members.append(cell)
        self.remove_reachable([cell], extensible=True)
        self.add_reachable(cell, reachable)

    def add_reachable(self, predecessor, cells):
        cells = set(cells) - set(self.members)  # exclude existing cells in cluster
        self.reachable |= cells
        for cell in cells:
            self.predecessor[cell] = predecessor

    def get_predecessor(self, cell):
        return self.predecessor[cell]

    def remove_reachable(self, cells, extensible=False):
        """
        有两种情况会用到这，一种是cell已经被别的簇占有，另一种是该cell从reachable晋升为extensible
        """
        for cell in cells:
            if not extensible:  # If extensible is True, no need to pop
                self.predecessor.pop(cell)
            assert cell in self.reachable, "not exist"
            self.reachable -= {cell}


class MyKmeans:
    """
    可以基于这个去做一些调整
    """

    def __init__(self, k, non_zero_idx, APSP_d, edge_list, max_iter=300):
        """
        :param k:
        :param non_zero_idx: get the correct cell number after removing obstacles
                            if non_zero_idx[0]=5, means the first non-obstacle cell is No.5 cell
        :param edge_list:  check whether this cell pair is adjacent
        :param max_iter:
        """
        self.k = k
        self.max_iter = max_iter
        self.features_count = -1
        self.clusters = []
        self.n_sample = None
        self.balanced_size = None
        self.non_zero_idx = non_zero_idx
        self.edge_list = edge_list
        self.APSP_d = APSP_d
        self.counter = None

    def fit_predict(self, data):
        """
        :param data: numpy数组，约定shape为：(数据数量，数据维度), 已排除空cell
        :type data: np.ndarray
        对于每个簇，都需要有一个生成树，每个节点都有自己的前驱节点
        distances: 表示distances[p]表示p点与所属根节点r之间的总路径长度，当不属于任何簇时，为-1
        """
        self.n_sample = data.shape[0]
        self.features_count = data.shape[1]
        self.balanced_size = np.ceil(self.n_sample / self.k)
        distances = {idx: -1 for idx in self.non_zero_idx}
        mapping = {cell_num: idx for idx, cell_num in enumerate(self.non_zero_idx)}
        # mapping[5]=2 means the No.5 cell is the 2nd non_zero cell

        # 等距离选择聚类中心
        for i in range(self.k):
            idx = self.non_zero_idx[self.n_sample//self.k*i]
            self.clusters.append(Cluster(idx))
            distances[idx] = 0
            self.clusters[i].add_reachable(idx, self.edge_list[idx])
        for i in range(self.max_iter):
            # 清空聚类
            while -1 in distances.values():
                for cluster in self.clusters:
                    to_remove = []  # 有些cell候补已经被别的簇拿走了，所以需要去掉
                    candidates = cluster.reachable  # reachable
                    min_d = float('inf')
                    min_idx = -1
                    for c in candidates:
                        if distances[c] != -1:  # 说明已经被别的簇吸收了
                            to_remove.append(c)
                            continue
                        pred = cluster.get_predecessor(c)
                        temp_d = distances[pred]+1
                        if temp_d < min_d:
                            min_d = temp_d
                            min_idx = c
                    if min_idx == -1:  # 说明已经没有候补了
                        continue
                    cluster.remove_reachable(to_remove)
                    cluster.add_member(min_idx, self.edge_list[min_idx])
                    distances[min_idx] = distances[cluster.get_predecessor(min_idx)] + 1

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
            if prev_centroids == cur_centroids:
                result = [-1]*self.n_sample
                print(f"after {i} iterations, centroids no longer change")
                for cluster_code, cluster in enumerate(self.clusters):
                    idxes = [mapping[c] for c in cluster.members]
                    for idx in idxes:
                        result[idx] = cluster_code
                assert -1 not in result
                # predecessors中有一部分是预备加入的点而不是已经加入簇的，需要排除
                for cluster in self.clusters:
                    cluster.predecessor = {key: value for key, value in cluster.predecessor.items()
                                           if key in cluster.members}
                return self.balanced_connect_processing(result)
            self.clusters = new_clusters
            distances = new_distances

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
        weight value of move=WM|Cr|+WM|Cs|-WM|Cr-pi|-WM|Cs+pi|
        ========
        执行move的条件：balanced value>0 or (balanced value=0 and weight value of move > 0)
        do move operation until no more remain
        """
        tolerance_size = 3
        self.counter = Counter(clustering)
        mapping = {cell_num: idx for idx, cell_num in enumerate(self.non_zero_idx)}
        # mapping[5]=2 means the No.5 cell is the 2nd non_zero cell
        for i in range(self.max_iter):
            for key, value in self.edge_list.items():
                idx1 = mapping[key]
                cls_1 = clustering[idx1]
                candidate = value
                for nxt_cell in candidate:
                    idx2 = mapping[nxt_cell]
                    cls_2 = clustering[idx2]
                    if cls_1 == cls_2:
                        continue
                    # 说明来自不同的cluster，是否分配需要进行连续性判断
                    cls_1_size, cls_2_size = self.counter[cls_1], self.counter[cls_2]
                    if cls_1_size > self.balanced_size > cls_2_size:
                        if self.feasibility_checking(key, cls_1):
                            self.assign_to_cluster(clustering, idx1, cls_1, cls_2)
                    elif cls_1_size - cls_2_size > tolerance_size:
                        if self.feasibility_checking(key, cls_1):
                            self.assign_to_cluster(clustering, idx1, cls_1, cls_2)
                    elif 0 <= cls_1_size - cls_2_size <= tolerance_size:
                        weight_value = abs(cls_1_size - self.balanced_size) + abs(cls_2_size - self.balanced_size) \
                                       - abs(cls_1_size - 1 - self.balanced_size) - abs(cls_2_size + 1 - self.balanced_size)
                        if weight_value > 0 and self.feasibility_checking(key, cls_1):
                            self.assign_to_cluster(clustering, idx1, cls_1, cls_2)
                    elif cls_1_size - cls_2_size < 0:
                        continue
        return clustering

    def feasibility_checking(self, idx, cls_1):
        print(f"checking {idx}")
        cur_cluster = deepcopy(self.clusters[cls_1])
        successors = [success for success, pred in cur_cluster.predecessor.items() if pred == idx]
        if successors:  # 有后继者，要保证其他后继者都能找到可连接的前驱节点
            # 没有后继者，分配给别的簇也不会有影响
            members = cur_cluster.members
            for suc in successors:
                for new_pred in members:
                    if new_pred == idx or suc == new_pred:
                        continue
                    a, b = suc, new_pred
                    if suc > new_pred:
                        a, b = b, a
                    if self.APSP_d[a, b] == 1 and cur_cluster.predecessor[new_pred] != suc:
                        # 相邻的其他节点选定为新的前驱节点,但不能是该节点的后继节点，会造成互相连接
                        cur_cluster.predecessor[suc] = new_pred
                        break
                else:
                    print(f"succ {suc} can't find new predecessor")
                    return False
        cur_cluster.members.remove(idx)
        cur_cluster.predecessor.pop(idx)
        self.clusters[cls_1] = cur_cluster
        return True

    def assign_to_cluster(self, clustering, idx, cls_1, cls_2):
        """
        分配完后还有一些东西需要更改，如cluster2需要为新来的点进行操作
        """
        clustering[idx] = cls_2
        cur_members = self.clusters[cls_2].members
        real_cell_code = self.non_zero_idx[idx]
        pred_idx = np.abs(np.array(cur_members)-real_cell_code).argmin()
        self.clusters[cls_2].predecessor[real_cell_code] = cur_members[pred_idx]
        self.clusters[cls_2].members.append(real_cell_code)
        self.counter[cls_1] -= 1
        self.counter[cls_2] += 1
        print(f"assign {(real_cell_code//21,real_cell_code%21)} from {cls_1} to {cls_2}")


