import torch

checkpoint = torch.load('../model/model_vae_100.tar')  # 先反序列化模型
start_epoch = checkpoint['epoch']
total_loss = checkpoint['loss']
print(start_epoch+1)
print(total_loss)
