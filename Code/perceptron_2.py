import numpy as np

"""
按照《统计学习方法（第二版）-李航》的书上的例题2.2 进行复现的
数据同例2.1，正样本点x_1 = (3,3)^T, x_2 = (4,3)^T，负样本点x_3 = (1,1)^T
试用感知机学习算法对偶形式求感知机模型。

很简单实现了例题，不过还是很粗糙，懒得整理。。。。。下次一定
"""

data = np.array([[3, 3, 1],
                 [4, 3, 1],
                 [1, 1, -1]])

X = data[:, 0:2]
y = data[:, 2]

alpha = np.zeros(len(X))
# print(alpha)

a = 0
b = 0
learning_rate = 1
N = len(X)
Gram = np.zeros([N, N])
# getGramMatrix


for i in range(N):
    for j in range(N):
        Gram[i][j] = X[i].dot(X[j])

print(Gram)

errorClassify = True
while errorClassify:
    errorClassify = False
    for i in range(N):
        sum_ = 0
        for j in range(N):
            # sum_ += np.array(alpha[j] * y[j] * X[j]).dot(X[i])
            # print(np.array(X[j]).dot(X[i]))
            sum_ += alpha[j] * y[j] * Gram[j][i]
        sum_ += b
        sum_ *= y[i]
        if sum_ <= 0:
            errorClassify = True
            alpha[i] = alpha[i] + learning_rate
            b = b + learning_rate * y[i]

print("得到结果：\nalpha：{0} "
      "\nb:{1}".format(alpha, b))
"""
得到结果：
alpha：[2. 0. 5.] 
b:-3
"""
