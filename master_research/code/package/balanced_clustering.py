from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
import math
from collections import Counter


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


class MyKmeans:
    """
    可以基于这个去做一些调整
    """

    def __init__(self, k, non_zero_idx, edge_list, tolerance=0.01, max_iter=300):
        """
        :param k:
        :param non_zero_idx: get the correct cell number after removing obstacles
                            if non_zero_idx[0]=5, means the first non-obstacle cell is No.5 cell
        :param edge_list:  check whether this cell pair is adjacent
        :param tolerance:
        :param max_iter:
        """
        self.k = k
        self.tol = tolerance
        self.max_iter = max_iter
        self.features_count = -1
        self.classifications = None
        self.centroids = None
        self.n_sample = None
        self.balanced_size = None
        self.non_zero_idx = non_zero_idx
        self.edge_list = edge_list

    def fit_predict(self, data):
        """
        :param data: numpy数组，约定shape为：(数据数量，数据维度)
        :type data: np.ndarray
        """
        self.n_sample = data.shape[0]
        self.features_count = data.shape[1]
        self.balanced_size = np.ceil(self.n_sample / self.k)
        # 初始化聚类中心（维度：k个 * features种数）
        self.centroids = np.zeros([self.k, data.shape[1]])
        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            # 清空聚类列表
            self.classifications = [[] for i in range(self.k)]
            # 对每个点与聚类中心进行距离计算
            result = []
            for feature_set in data:
                # 预测分类
                classification = self.predict(feature_set)
                # 加入类别
                self.classifications[classification].append(feature_set)
                result.append(classification)
            # 记录前一次的结果
            prev_centroids = np.ndarray.copy(self.centroids)

            # 更新中心
            for classification in range(self.k):
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            # 检测相邻两次中心的变化情况
            for c in range(self.k):
                if np.linalg.norm(prev_centroids[c] - self.centroids[c]) > self.tol:
                    break

            # 如果都满足条件（上面循环没break），则返回
            else:
                return result

    def predict(self, data):
        # 距离
        distances = np.linalg.norm(data - self.centroids, axis=1)  # euclidean metric
        # linear scale
        # scale = []
        # for i in range(self.k):
        #     if cur_clus_size[i] < 50:
        #         scale.append(0.8)
        #     else:
        #         scale.append(0.01*cur_clus_size[i]+0.18)
        # distances = distances*np.array(scale)   # 后期距离差的很多，缩放效果有限
        # print("current_size_of_cluster", list(map(len, self.classifications)))
        # print("after scale:", distances)
        # print("="*100)
        return distances.argmin()

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
        for key, value in self.edge_list.items():
            idx1 = mapping[key]
            cls_1 = clustering[idx1]
            candidate = value
            for nxt_cell in candidate:
                idx2 = mapping[nxt_cell]
                cls_2 = clustering[idx2]
                if cls_1 == cls_2:
                    continue
                cls_1_size, cls_2_size = self.counter[cls_1], self.counter[cls_2]
                if cls_1_size > self.balanced_size > cls_2_size:
                    # 是否需要一个check函数，check连通性
                    self.assign_to_cluster(clustering, idx1, cls_1, cls_2)
                elif cls_1_size - cls_2_size > tolerance_size:
                    self.assign_to_cluster(clustering, idx1, cls_1, cls_2)
                elif 0 <= cls_1_size - cls_2_size <= tolerance_size:
                    weight_value = abs(cls_1_size - self.balanced_size) + abs(cls_2_size - self.balanced_size) \
                                   - abs(cls_1_size - 1 - self.balanced_size) - abs(cls_2_size + 1 - self.balanced_size)
                    if weight_value > 0:
                        self.assign_to_cluster(clustering, idx1, cls_1, cls_2)
                elif cls_1_size - cls_2_size < 0:
                    continue
        return clustering

    def assign_to_cluster(self, clustering, idx, cls_1, cls_2):
        clustering[idx] = cls_2
        self.counter[cls_1] -= 1
        self.counter[cls_2] += 1
        print(f"assign {idx} from {cls_1} to {cls_2}")
