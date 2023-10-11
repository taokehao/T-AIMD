#!/usr/bin/env python3
# encoding: utf-8
"""
@Time    : 2023/4/20
@Author  : Kehao Tao
@File    : regression_model.py
@Software: PyCharm
@desc: catboost 回归模型预测（特征：transformer的encoder部分）
"""
from xgboost.sklearn import XGBRegressor
import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor


def plot_regression(model, x_data, y_data, num):
    # kf = KFold(n_splits=num, shuffle=True, random_state=1)
    kf = KFold(n_splits=num, shuffle=False)
    # kf = KFold(n_splits=num, shuffle=True, random_state=4)
    mae_total = 0
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
    print("平均MAE:", mae_total / num)
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
                    print("est,lea,dep,sub：", est, lea, dep, sub, '  catboost结果为：')
                    clf = CatBoostRegressor(iterations=1200,
                                            learning_rate=0.020000000000000004,
                                            max_depth=8,
                                            subsample=7
                                            )
                    # clf = CatBoostRegressor(iterations=1500,
                    #                         learning_rate=0.03,
                    #                         l2_leaf_reg=3,
                    #                         bagging_temperature=1,
                    #                         subsample=0.66,
                    #                         random_strength=1,
                    #                         depth=6,
                    #                         rsm=1,
                    #                         one_hot_max_size=2,
                    #                         leaf_estimation_method='Gradient',
                    #                         fold_len_multiplier=2,
                    #                         border_count=128)
                    now_res = plot_regression(clf, x_data, y_data, num)
                    if now_res < min_res:
                        min_res = now_res
                        max_est = est
                        max_lea = lea
                        max_dep = dep
                        max_sub = sub
                    print("此时搜索的最大准确率：", min_res, "  此时的est,lea,dep,sub：", max_est, max_lea, max_dep,
                          max_sub)
                    sub += 0.001
                dep += 1
            lea += 0.001
        est += 1

    print("")
    print("搜索的最大准确率：", min_res, "  此时的est,lea,dep,sub：", max_est, max_lea, max_dep, max_sub)
