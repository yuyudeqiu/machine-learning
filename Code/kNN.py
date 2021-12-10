"""
线性扫描的KNN算法
"""
import numpy as np


def L2(x_1, x_2):
    """
    计算欧式距离
    :param x_1:
    :param x_2:
    :return:返回两点之间的距离
    """
    return np.linalg.norm(np.array(x_1) - np.array(x_2))


def KNN(k: int, data, x):
    # 距离列表
    ops = []
    label = []

    labels = data[:, -1].tolist()
    # 计算距离
    for d in data[:, :-1]:
        ops.append(L2(d, x))
    print(ops)
    for i in range(k):
        max_index = ops.index(max(ops))
        label.append(labels[max_index])
        ops.pop(max_index)
        labels.pop(max_index)
    print(label)
    maxLable = max(label, key=label.count)
    return maxLable


if __name__ == '__main__':
    data = np.array([
        [1, 1, 0],
        [2, 1, 0],
        [4, 4, 1],
        [4, 5, 1],
        [0, 1, 0]
    ])
    print(KNN(3, data, [0, 0]))
