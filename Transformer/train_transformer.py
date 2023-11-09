#!/usr/bin/env python3
# encoding: utf-8
"""
@Time    : 2023/4/11
@Author  : Kehao Tao
@File    : train_transformer.py
@Software: PyCharm
@desc: transformer 浮点数序列复制
"""
import sys

sys.path.append("../")
import torch
from torch import nn
from torch.autograd import Variable
from transformer import TransformerTS
import numpy as np
from myfunction import draw_loss, generate_data, output_feature, output_loss
import logging

"""
设置使用的设备
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('You are using: ' + str(device))

"""
存储中间encoder的特征
"""
feature_encoder = []

"""
全局变量设置
"""
initial_dim = 100  # 输入序列的长度
middle_dim = 100
batch_num = 4  # 每一组的数据数
dim_feedforward = 256  # transformer中间层数
total_epoch = 10000  # 总共训练的轮数

"""
导入训练数据
"""
train_data = generate_data(initial_dim, batch_num)

"""
模型参数设置
"""
model = TransformerTS(initial_dim=initial_dim,
                      middle_dim=middle_dim,
                      d_model=6,  # 编码器/解码器输入中预期特性的数量
                      nhead=2,
                      num_encoder_layers=3,
                      num_decoder_layers=3,
                      dim_feedforward=dim_feedforward,
                      dropout=0.1).to(device)  # 设置模型参数
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # 设置模型的优化器
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1)  # Learning Rate Decay
criterion = nn.MSELoss()  # 设置模型的损失函数
train_loss_list = []  # 每次epoch的loss保存起来
total_loss = 31433357277  # 网络训练过程中最大的loss

'''
记录实验中的loss值的设置
'''
handlers = [logging.FileHandler('../data/logging_of_loss_200.txt', encoding="utf-8"),
            logging.StreamHandler()]
logging.basicConfig(level=logging.INFO,
                    datefmt='20%y-%m-%d %H:%M',
                    format='%(asctime)s:%(message)s',
                    handlers=handlers)

'''
训练模型的函数
'''
def train_transformer(epoch):
    """ 训练模型
    :param epoch: 训练的轮数
    """
    global total_loss, feature_encoder
    feature_encoder_epoch = []
    mode = True
    model.train(mode=mode)  # 模型设置为训练模式
    loss_epoch = 0  # 一次epoch的loss总和
    for idx, sin_input in enumerate(train_data):
        sin_input_np = sin_input.numpy()  # 1D
        cos_output = sin_input.contiguous()

        sin_input_torch = Variable(torch.from_numpy(sin_input_np[np.newaxis, :]))  # 3D

        prediction, feature_encoder_in_epoch = model(sin_input_torch.to(device))  # torch.Size([batch size])
        loss = criterion(prediction, cos_output.to(device))  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # back propagation, compute gradients
        optimizer.step()  # apply gradients
        # scheduler.step()
        # print(scheduler.get_lr())

        loss_epoch += loss.item()  # 将每个batch的loss累加，直到所有数据都计算完毕
        for i in feature_encoder_in_epoch.tolist():
            feature_encoder_epoch.append(i)
        if idx == len(train_data) - 1:
            # print('Train Epoch:{}\tLoss:{:.9f}'.format(epoch, loss_epoch / 97))
            logging.info('Train Epoch:{}\tLoss:{:.9f}'.format(epoch, loss_epoch))
            train_loss_list.append(loss_epoch)
            if loss_epoch < total_loss:
                total_loss = loss_epoch
                # torch.save(cal-pre-data, '..\\cal-pre-data\\model97-2.pkl')  # save cal-pre-data
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_loss
                }, '..\\model\\model200.tar')  # save cal-pre-data
                feature_encoder = feature_encoder_epoch.copy()
                feature_encoder_epoch = []


if __name__ == '__main__':
    # 模型训练
    print("Start Training...")
    for i in range(total_epoch):
        train_transformer(i)
        # print(feature_encoder[0])
    print("Stop Training!")
    print('最小loss值是：', total_loss)
    # 绘制loss曲线
    draw_loss(train_loss_list, total_epoch)
    # 存储loss数据到文件中
    output_loss(train_loss_list)
    # 存储encoder的特征到csv文件
    output_feature(feature_encoder)
