import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from Transformer.myfunction import draw_loss, generate_data, output_feature, output_loss
import logging
import csv

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
criterion = nn.MSELoss()  # 设置模型的损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

handlers = [logging.FileHandler('../data/vae_loss_100.txt', encoding="utf-8"),
            logging.StreamHandler()]
logging.basicConfig(level=logging.INFO,
                    datefmt='20%y-%m-%d %H:%M',
                    format='%(asctime)s:%(message)s',
                    handlers=handlers)
total_loss = 31433357277  # 网络训练过程中最大的loss

# 开始训练
for epoch in range(num_epochs):
    feature_encoder_epoch = []
    mode = True
    model.train(mode=mode)  # 模型设置为训练模式
    loss_epoch = 0  # 一次epoch的loss总和
    for idx, x in enumerate(train_data):
        # 前传
        x_reconst, mu, log_var, feature_encoder_in_epoch = model(x.to(device))

        # 计算重建损失和kl散度
        reconst_loss = criterion(x_reconst, x.to(device))
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # 反向传播和优化
        loss = reconst_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        for i in feature_encoder_in_epoch.tolist():
            feature_encoder_epoch.append(i)
        if idx == len(train_data) - 1:
            # print('Train Epoch:{}\tLoss:{:.9f}'.format(epoch, loss_epoch / 97))
            logging.info('Train Epoch:{}\tLoss:{:.9f}'.format(epoch+1, loss_epoch))

            if loss_epoch < total_loss:
                total_loss = loss_epoch
                # torch.save(model, '..\\model\\model97-2.pkl')  # save model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_loss
                }, '..\\model\\model_vae_100.tar')  # save model
                feature_encoder = feature_encoder_epoch.copy()
                feature_encoder_epoch = []


# 存储encoder的特征到csv文件
for i in feature_encoder:
    csvFile = open("../data/feature_encoder_vae100.csv", 'a', newline='', encoding='utf-8')
    writer = csv.writer(csvFile)
    writer.writerow(i)  # 数据写入文件中zz
    csvFile.close()
