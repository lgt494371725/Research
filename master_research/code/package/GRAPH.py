# 邻接表法
class Vertex:
    def __init__(self, key):
        self.id = key
        self.connectedTo = {}

    def addNeighbor(self, nbr, weight=0):  # nbr是顶点实例作为字典的key
        self.connectedTo[nbr] = weight

    def __str__(self):  # 需要打印信息时弹出的魔法方法
        return str(self.id) + '  connectedTo:' \
               + str([x.id for x in self.connectedTo])  # x是顶点对象，作为key

    def getConnections(self):
        return self.connectedTo.keys()

    def getid(self):
        return self.id

    def getweight(self, nbr):
        return self.connectedTo[nbr]

    def change_weight(self, nbr, weight):
        self.connectedTo[nbr] = weight


class Graph:  # 创建的是有向图
    def __init__(self):
        self.verlist = {}
        self.numver = 0

    def addVertex(self, key):
        self.numver += 1
        newvertex = Vertex(key)
        self.verlist[key] = newvertex  # 把类对象作为字典的value
        return newvertex

    def getvertex(self, n):
        if n in self.verlist:
            return self.verlist[n]
        else:
            return None

    def __contains__(self, n):
        return n in self.verlist

    def addEdge(self, f, t, weight=0):
        if f not in self.verlist:  # 如果点不存在，就先添加点
            nv = self.addVertex(f)
        if t not in self.verlist:
            nv = self.addVertex(t)
        self.verlist[f].addNeighbor(self.verlist[t], weight)
        # self.verlist[t].addNeighbor(self.verlist[f], weight)

    def change_edge_w(self, f, t, weight):
        self.verlist[f].changeweight(self.verlist[t], weight)
        # self.verlist[t].changeweight(self.verlist[f], weight)

    def getVers(self):
        return self.verlist.keys()

    def __iter__(self):  # 返回的是顶点实例，然后会触发顶点中的__str__方法
        return iter(self.verlist.values())
