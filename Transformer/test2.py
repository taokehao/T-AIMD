# import sys
#
# sys.path.append("../")
# import torch
# from torch.autograd import Variable
# import numpy as np
# from myfunction import generate_data, generate_test
# import csv
# from transformer import TransformerTS
# from torch import nn
#
# checkpoint = torch.load('..\\cal-pre-data\\model600.tar')  # 先反序列化模型
# total_loss = checkpoint['loss']
# epoch = checkpoint['epoch']
# print(total_loss, epoch)
#
# for i in range(1, 10001):
#     lie = [i]
#     csvFile = open("./jishu.csv", 'a', newline='', encoding='utf-8')
#     writer = csv.writer(csvFile)
#     writer.writerow(lie)  # 数据写入文件中zz
#     csvFile.close()


# #!/usr/bin/env python3
# # encoding: utf-8
# """
# @Time    : 2023/4/11
# @Author  : Kehao Tao
# @File    : test_transformer.py
# @Software: PyCharm
# @desc: transformer 测试
# """
#
# import sys
#
# sys.path.append("../")
# import torch
# from torch.autograd import Variable
# import numpy as np
# from myfunction import generate_data, generate_test
# import csv
# from transformer import TransformerTS
# from torch import nn
#
# """
# 设置使用的设备
# """
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # device = torch.device("cpu")
# print('You are using: ' + str(device))
#
# """
# 存储中间encoder的特征
# """
# feature_encoder = []
#
# """
# 全局变量设置
# """
# initial_dim = 100  # 输入序列的长度
# middle_dim = 100
# batch_num = 4  # 每一组的数据数
# dim_feedforward = 256  # transformer中间层数
# total_epoch = 10000  # 总共训练的轮数
#
# """
# 导入训练数据
# """
# # train_data = generate_data(initial_dim, batch_num)
# test_data = generate_test(initial_dim, batch_num, data_num=1)
#
# """
# 模型参数设置
# """
# net = TransformerTS(initial_dim=initial_dim,
#                     middle_dim=middle_dim,
#                     d_model=6,  # 编码器/解码器输入中预期特性的数量
#                     nhead=3,
#                     num_encoder_layers=3,
#                     num_decoder_layers=3,
#                     dim_feedforward=dim_feedforward,
#                     dropout=0.1).to(device)  # 设置模型参数
#
# checkpoint = torch.load('..\\cal-pre-data\\model600.tar')  # 先反序列化模型
# net.load_state_dict(checkpoint['model_state_dict'])
# net.eval()
# prediction_ouput = []
#
# for idx, sin_input in enumerate(test_data):
#     sin_input_np = sin_input.numpy()  # 1D
#     cos_output = sin_input.contiguous()
#
#     sin_input_torch = Variable(torch.from_numpy(sin_input_np[np.newaxis, :]))  # 3D
#
#     prediction, feature_encoder_in_epoch = net(sin_input_torch.to(device))  # torch.Size([batch size])
#     for i in prediction.tolist():
#         prediction_ouput.append(i)
#     if idx == len(test_data) - 1:
#         feature_encoder = prediction_ouput.copy()
#
# for i in feature_encoder:
#     csvFile = open("./test_ouput.csv", 'a', newline='', encoding='utf-8')
#     writer = csv.writer(csvFile)
#     writer.writerow(i)  # 数据写入文件中zz
#     csvFile.close()

import numpy as np
import csv
y_data = np.log10(np.loadtxt("../data/y_data.csv", delimiter=',')).tolist()
for i in y_data:
    newrow = [i]
    csvFile = open("./test_ouput.csv", 'a', newline='', encoding='utf-8')
    writer = csv.writer(csvFile)
    writer.writerow(newrow)  # 数据写入文件中zz
    csvFile.close()
