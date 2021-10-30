"""
自己使用 KDTree 实现的 KNN 算法模型
"""

import numpy as np
import heapq


class Node:
    """
    存储一条数据的节点
    """
    def __init__(self, data, dim_index, left=None, right=None):
        """
        :param data: 存储的数据
        :param dim_index: 结点分支依据的属性索引
        :param left: 左孩子
        :param right: 右孩子
        """
        self.data = data
        self.dim_index = dim_index
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.data[0] < other.data[0]


class MyKNNClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.n_dots = 0   # 总数据量
        self.n_features = 0  # 属性个数
        self.X, self.Y = [], []
        self.label_set = set()  # 标签集合
        self.x_to_y = dict()  # 依据给定的数据（必须是训练集中包含的），得到数据的标签
        self.root = None  # 根节点

    def fit(self, X, Y):
        X = np.array(X)
        Y = np.array(Y).flatten()
        assert X.shape[0] == Y.shape[0]
        self.n_dots = len(X)
        self.n_features = X.shape[1]
        self.X = X
        self.Y = Y
        self.label_set = set(self.Y)

        # 以哈希表的形式存储 X 和 Y 的对应关系
        for i in range(self.n_dots):
            self.x_to_y[hash(tuple(X[i]))] = Y[i]

        def create(X, dim_index):
            if len(X) == 0:
                return None
            # 依据第 dim_index 列属性进行排序
            X = sorted(X, key=lambda x: x[dim_index])
            mid = len(X) // 2

            # 递归构造 KDTree
            left_child = create(X[:mid], (dim_index + 1) % self.n_features)
            right_child = create(X[mid + 1:], (dim_index + 1) % self.n_features)

            return Node(X[mid], dim_index, left_child, right_child)

        self.root = create(X, 0)

    def _search(self, x):
        dots_set = [(-np.inf, None)] * self.n_neighbors

        def visit(node):
            # 递归查找
            if node:
                dis = x[node.dim_index] - node.data[node.dim_index]
                visit(node.left if dis < 0 else node.right)
                curr_dis = np.linalg.norm(x - node.data)
                heapq.heappushpop(dots_set, (-curr_dis, node))
                if -(dots_set[0][0]) > abs(dis):
                    visit(node.right if dis < 0 else node.left)

        visit(self.root)  # 递归查找
        largest_dots_set = heapq.nlargest(self.n_neighbors, dots_set)
        return np.array([i[1].data for i in largest_dots_set])

    def predict(self, X):
        X = np.array(X)
        Y = []

        progress = 1

        for x in X:
            # 训练进度（考虑到算法效率太低，可能需要几十分钟……
            print('\r %.2f%%' % (progress / X.shape[0] * 100), end='')
            progress += 1

            # 查找距离最近的 n_neighbors 个点
            dots_set = self._search(x)
            # 根据数据点得到标签
            labels = [self.x_to_y[hash(tuple(dot))] for dot in dots_set]

            # 统计频率最高的标签
            count = dict(zip(self.label_set, [0] * self.n_dots))
            for label in labels:
                count[label] += 1
            label = sorted(count.items(), key=lambda d: d[1], reverse=True)[0][0]

            Y.append(label)

        return np.array(Y)


if __name__ == "__main__":
    X = np.array([[2, 3, 1], [5, 4, 2], [9, 6, 3], [4, 7, 4], [8, 1, 5], [7, 2, 6]])
    Y = np.array([0, 1, 0, 1, 0, 1])
    model = MyKNNClassifier(1)
    model.fit(X, Y)

    print(model.predict(X))
