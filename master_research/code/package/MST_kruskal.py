# 最小生成数（克鲁斯卡尔法）


class UnionFindSet:
    """
    开始有n个元素，每个集合的代表节点都是本身
    记录每个集合的深度，最初都是1
    适用于字母作为key
    """
    def __init__(self, fa=None):
        self.fa = {i: i for i in fa}
        self.rank = {i: 1 for i in self.fa.keys()}

    def find(self, x):   # 查找x所属集合的代表节点
        if self.fa[x] != x:
            self.fa[x] = self.find(self.fa[x])  # 将集合内所属节点的指向都设置为代表节点
        return self.fa[x]

    def union(self, i, j):  # 将i集合合并到j上
        x = self.find(i)
        y = self.find(j)
        if x != y:  # 相等则说明来自同一集合，无需操作
            if self.rank[x] <= self.rank[y]:  # 把简单的树往复杂的树上合并，这样合并后到根节点距离变长的节点个数比较少。
                self.fa[x] = y
            else:
                self.fa[y] = x
            if self.rank[x] == self.rank[y]:
                self.rank[y] += 1


def MiniSpanTree_kruskal(g):
    result = 0
    # 首先构造边集数组并排序
    Edgelist = getedgelist(g)
    # 使用并查集方法查看是否构成闭环
    # 初始将所有顶点单独处于一个集合，每次加入一条边，判断开始点和终点是否处于一个集合
    # 若是，则会构成闭环
    searchset = UnionFindSet([V.id for V in g])  # id是字母，并查集是字典类型
    choice = []
    for edge in Edgelist:
        head = searchset.find(edge[0])
        tail = searchset.find(edge[1])
        if head != tail:  # 不在一连通分支，则可用
            searchset.union(head, tail)  # 合并两个连通分支
            choice.append(edge)
            result += edge[-1]
    return choice, result


def getedgelist(g):
    alist = []
    for V in g:
        for W in V.getConnections():
            alist.append([V.id, W.id, V.getweight(W)])
    alist.sort(key=lambda x: x[-1], reverse=False)
    return alist


# g = Graph()
# g.addEdge('a', 'c', 3)
# g.addEdge('a', 'b', 1)
# g.addEdge('c', 'b', 2)
# g.addEdge('c', 'd', 4)
# g.addEdge('c', 'e', 4)
# g.addEdge('d', 'e', 2)
# g.addEdge('d', 'f', 3)
# g.addEdge('b', 'd', 5)
# g.addEdge('e', 'f', 5)
# choice, result = MiniSpanTree_kruskal(g)
# print(choice)
# print(result)
