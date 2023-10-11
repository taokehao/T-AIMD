import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import csv
import numpy as np


def generate_data(initial_dim, batch_num):
    train_data = []
    total_data = np.loadtxt('../data/raw_data.csv', delimiter=',', dtype=float)
    total_data = total_data[:, :initial_dim]
    for i in range(int(388/batch_num)):
        train_data.append(total_data[i*batch_num:i*batch_num+batch_num, :].tolist())
    train_data = torch.FloatTensor(train_data)
    return train_data


# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('You are using: ' + str(device))


# 超参数
image_size = 100
h_dim = 200
z_dim = 50
num_epochs = 10000
learning_rate = 1e-3
initial_dim = 100  # 输入序列的长度
batch_num = 4  # 每一组的数据数

"""
导入训练数据
"""
train_data = generate_data(initial_dim, batch_num)

"""
存储中间encoder的特征
"""
feature_encoder = []

# VAE模型
class VAE(nn.Module):
    def __init__(self, image_size=100, h_dim=200, z_dim=50):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(image_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, image_size)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc4(z))
        return F.sigmoid(self.fc5(h))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var, z


model = VAE().to(device)
checkpoint = torch.load('../model/model_vae.tar')  # 先反序列化模型
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
prediction_ouput = []

for idx, x in enumerate(train_data):
    # 前传
    x_reconst, mu, log_var, feature_encoder_in_epoch = model(x.to(device))

    for i in x_reconst.tolist():
        prediction_ouput.append(i)
    if idx == len(train_data) - 1:
        feature_encoder = prediction_ouput.copy()

for i in feature_encoder:
    csvFile = open("./vae_100_ouput.csv", 'a', newline='', encoding='utf-8')
    writer = csv.writer(csvFile)
    writer.writerow(i)  # 数据写入文件中zz
    csvFile.close()
