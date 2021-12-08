import numpy as np

"""
按照《统计学习方法（第二版）-李航》的书上的例题2.1进行的
正点是x_1 = (3,3)^T, x_2 = (4,3)^T，负点是x_3 = (1,1)^T
试用感知机学习算法的原始形式求感知机模型 f(x)=sign(wx+b)

得到了和书本一样的w和b，不过写的很粗糙，也没有做封装，有空一定搞。。。。
先这样
"""

data = np.array([[3, 3, 1],
                 [4, 3, 1],
                 [1, 1, -1]])

X = data[:, 0:2]
y = data[:, 2]
print("X", X)
print("y", y)

w = np.array([0, 0])
b = 0
learning_rate = 1
print("w", w)
errorClassify = True
print("==================================")

while errorClassify:
    errorClassify = False
    for i in range(len(X)):
        if y[i] * (w.dot(X[i]) + b) <= 0:
            errorClassify = True
            for j in range(len(X[i])):
                w[j] = w[j] + learning_rate * y[i] * X[i][j]
            b = b + learning_rate * y[i]
        print("w: {0} b: {1}".format(w, b))

print("求得 w:{0} b:{1}".format(w, b))


def sign(n):
    if n > 0:
        return 1
    else:
        return -1


test_X = np.array([4, 4])
print("预测：", test_X, "的类别：", end=" ")  # 求得 w:[1 1] b:-3
print(sign(w.dot(test_X) + b))  # 预测： [0 0] 的类别： -1
