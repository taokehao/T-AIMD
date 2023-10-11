#!/usr/bin/env python3
# encoding: utf-8
"""
@Time    : 2023/4/20
@Author  : Kehao Tao
@File    : regression_model.py
@Software: PyCharm
@desc: catboost 回归模型预测（特征：原始的951维特征）
"""
from xgboost.sklearn import XGBRegressor
from sklearn import metrics
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor
import numpy as np
import matplotlib.pyplot as plt


def plot(a, b):
    for i in a:
        print(i)
    for i in b:
        print(i)
    x_label = []
    for i in range(1,389):
        x_label.append(i)
    plt.scatter(x_label, a, c='r')
    plt.scatter(x_label, b, c='b')
    plt.show()


def plot_regression(model, x_data, y_data, num):
    # kf = KFold(n_splits=num, shuffle=True, random_state=1)
    # kf = KFold(n_splits=num, shuffle=True, random_state=4)
    kf = KFold(n_splits=num, shuffle=False)
    mae_total = 0
    y_pre_total = []
    y_real_total = []
    for train, test in kf.split(x_data):
        # print(train, test)
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        for i in train:
            x_train.append(x_data[i])
            y_train.append(y_data[i])
        for j in test:
            x_test.append(x_data[j])
            y_test.append(y_data[j])
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test)
        model.fit(x_train, y_train)
        # 进行预测
        y_pred = model.predict(x_test)
        # 展示预测结果
        # print("预测值:")
        # print(y_pred)
        # print("真实值:")
        # print(y_test)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        print("MAE", mae)
        print("--------")
        mae_total += mae
        for idx in y_pred:
            y_pre_total.append(idx)
        for idx in y_test:
            y_real_total.append(idx)
    print("平均MAE:", mae_total / num)
    print(len(y_real_total))
    plot(y_real_total, y_pre_total)
    return mae_total / num


if __name__ == '__main__':
    # readlines = open("./element_features.txt", 'r').readlines()
    # for row in readlines:
    #     data = open('./element_features2.txt', 'a')
    #     print(row.replace("  ", "    "), file=data)
    #     data.close()

    x_data = np.loadtxt("../data/feature_encoder_600+material.csv", delimiter=',').tolist()
    y_data = np.log10(np.loadtxt("../data/y_data.csv", delimiter=',')).tolist()
    # y_data = np.loadtxt("../data/y_data.csv", delimiter=',').tolist()

    # XGboost
    print("请输入KFlod的值:", end='')
    num = int(input())

    min_res = 100
    max_est = 0
    max_lea = 0
    max_dep = 0
    max_sub = 0
    est = 35
    lea = 0.3
    dep = 3
    sub = 0.75
    while est <= 50:
        lea = 0.3
        while lea <= 0.4:
            dep = 3
            while dep <= 10:
                sub = 0.75
                while sub <= 0.9:
                    print("est,lea,dep,sub：", est, lea, dep, sub, '  GDBC结果为：')
                    clf = CatBoostRegressor(iterations=1200,
                                            learning_rate=0.020000000000000004,
                                            max_depth=8,
                                            subsample=7
                                            )
                    now_res = plot_regression(clf, x_data, y_data, num)
                    if now_res < min_res:
                        min_res = now_res
                        max_est = est
                        max_lea = lea
                        max_dep = dep
                        max_sub = sub
                    print("此时搜索的最小MAE：", min_res, "  此时的est,lea,dep,sub：", max_est, max_lea, max_dep,
                          max_sub)
                    sub += 0.001
                dep += 1
            lea += 0.001
        est += 1

    print("")
    print("搜索的最小MAE：", min_res, "  此时的est,lea,dep,sub：", max_est, max_lea, max_dep, max_sub)
