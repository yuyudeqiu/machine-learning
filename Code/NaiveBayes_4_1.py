"""
《统计学习方法（第二版）》 - 李航 例题4.1的代码复现

例4.1 试由表4.1的训练数据学习一个朴素贝叶斯分类器并确定x=(2, S)^T的类标记y。表X^{(1)}，X^{(2)}为特征，
取值的集合分别为 A_1 = {1, 2, 3}, A_2 = {S, M, L}, Y为类标记， Y ∈ C = {1, -1}
"""
import math

import numpy as np
import pandas as pd


def NaiveBayes(train_data: pd.DataFrame):
    parameters = {}
    features = train_data.columns.values[:-1]
    label = train_data.columns.values[-1]
    parameters[label] = {}
    label_values = train_data[label].unique()
    for val in label_values:
        D_c = len(train_data[train_data[label] == val])
        D = len(train_data)
        N = len(label_values)
        parameters[label][val] = D_c / D
        # 计算概率时避免分母为0的情况，加入拉普拉斯平滑
        # parameters[label][val] = (D_c + 1) / (D + N)

    for feat in features:
        # print(feat)
        if feat not in parameters.keys():
            parameters[feat] = {}
        if ('object' in str(train_data[feat].dtype)) or ('int' in str(train_data[feat].dtype)):
            feat_values = train_data[feat].unique()
            N_i = len(feat_values)

            for feat_val in feat_values:

                parameters[feat][feat_val] = {}
                for label_val in label_values:
                    D_ci = len(train_data[train_data[label] == label_val][train_data[feat] == feat_val])
                    D_c = len(train_data[train_data[label] == label_val])
                    # parameters[feat][feat_val][label_val] = (D_ci + 1) / (D_c + N_i) # 计算概率时避免分母为0的情况，加入拉普拉斯平滑
                    parameters[feat][feat_val][label_val] = D_ci / D_c
        else:
            for label_val in label_values:
                parameters[feat][label_val] = {}
                mean_ci = train_data[train_data[label] == label_val][feat].mean()
                std_ci = train_data[train_data[label] == label_val][feat].std()
                parameters[feat][label_val]['mean'] = mean_ci
                parameters[feat][label_val]['std'] = std_ci

    return parameters


def norm_pdf(val, mean, std):
    pdf = 1 / (math.sqrt(2 * math.pi) * std) * math.exp(-math.pow(val - mean, 2) / (2 * std * std))
    return pdf


def predict(X: pd.DataFrame, parameters: {}):
    features = X.columns.values
    label = "y"
    label_values = ['-1', '1']
    res = ''
    max_p = -1

    for label_val in label_values:
        p = parameters[label][label_val]
        for feat in features:
            if ('object' in str(X[feat].dtype)) or ('int' in str(X[feat].dtype)):
                p *= parameters[feat][str(X.loc[0, feat])][label_val]
            else:
                p *= norm_pdf(X.loc[0, feat], parameters[feat][label_val]['mean'], parameters[feat][label_val]['std'])
        if p > max_p:
            res = label_val
            max_p = p

        print(res, ' ', p)

    return res, max_p


if __name__ == '__main__':
    dataSet = np.array([
        [1, "S", -1],
        [1, "M", -1],
        [1, "M", 1],
        [1, "S", 1],
        [1, "S", -1],
        [2, "S", -1],
        [2, "M", -1],
        [2, "M", 1],
        [2, "L", 1],
        [2, "L", 1],
        [3, "L", 1],
        [3, "M", 1],
        [3, "M", 1],
        [3, "L", 1],
        [3, "L", -1],
    ])
    dataSet = pd.DataFrame(dataSet)

    dataSet.columns = ["x1", "x2", "y"]
    parameters = NaiveBayes(dataSet)
    print(parameters)
    x = pd.DataFrame([[3, "L"]])
    x.columns = ["x1", "x2"]
    print(x)
    predict(x, parameters)
