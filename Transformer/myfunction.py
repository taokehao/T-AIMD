#!/usr/bin/env python3
# encoding: utf-8
"""
@Time    : 2023/4/11
@Author  : Kehao Tao
@File    : myfunction.py
@Software: PyCharm
@desc: 一些自定义的函数
"""
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch
import random
import csv
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# 导入数据集的类
class MyDataset(Dataset):
    def __init__(self, csv_file):
        self.lines = open(csv_file).readlines()

    def __getitem__(self, index):
        # 获取索引对应位置的一条数据
        cur_line = self.lines[index].split(',')

        sin_input = np.float32(cur_line[1].strip())
        cos_output = np.float32(cur_line[2].strip())

        return sin_input, cos_output

    def __len__(self):
        return len(self.lines)  # MyDataSet的行数


def draw_loss(Loss_list,epoch):
    """绘制loss曲线
    :param Loss_list: 每一轮的loss数据
    :param epoch: 训练轮数
    """
    y_train_loss = Loss_list  # loss值，即y轴
    x_train_loss = []
    for i in range(epoch):
        x_train_loss.append(i+1)

    plt.figure()

    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('iters')  # x轴标签
    plt.ylabel('loss')  # y轴标签

    # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
    # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
    plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="train loss")
    plt.legend()
    plt.title('Loss curve')
    plt.show()


# def generate_data(initial_dim):
#     batch_data = []
#     train_data = []
#     for curve in range(1, 101):
#         data_list = []
#         rand_num = random.uniform(1, 10)
#         rand_num_dis = random.uniform(1, 10)
#         for i in range(initial_dim):
#             data_list.append(i * rand_num + rand_num_dis)
#         batch_data.append(data_list)
#         if curve % 10 == 0:
#             train_data.append(batch_data)
#             batch_data = []
#         print(data_list)
#     train_data = torch.FloatTensor(train_data)
#
#     return train_data

def generate_data(initial_dim, batch_num):
    train_data = []
    total_data = np.loadtxt('../data/raw_data.csv', delimiter=',', dtype=float)
    total_data = total_data[:, :initial_dim]
    for i in range(int(388/batch_num)):
        train_data.append(total_data[i*batch_num:i*batch_num+batch_num, :].tolist())
    train_data = torch.FloatTensor(train_data)
    return train_data

def output_feature(feature_encoder):
    # 将最终数据写入csv文本
    for i in feature_encoder:
        csvFile = open("../data/feature_encoder_200.csv", 'a', newline='', encoding='utf-8')
        writer = csv.writer(csvFile)
        writer.writerow(i)  # 数据写入文件中zz
        csvFile.close()

def output_loss(train_loss_list):
    # 将最终数据写入csv文本
    for i in train_loss_list:
        newRow = []
        newRow.append(i)
        csvFile = open("../data/train_loss_list_200.csv", 'a', newline='', encoding='utf-8')
        writer = csv.writer(csvFile)
        writer.writerow(newRow)  # 数据写入文件中zz
        csvFile.close()

def generate_test(initial_dim, batch_num, data_num):
    teat_data = []
    total_data = np.loadtxt('../test_data/tinghua/test_data.csv', delimiter=',', dtype=float)
    total_data = total_data[:, :initial_dim]
    for i in range(int(data_num / batch_num)):
        teat_data.append(total_data[i * batch_num:i * batch_num + batch_num, :].tolist())
    teat_data = torch.FloatTensor(teat_data)
    return teat_data
