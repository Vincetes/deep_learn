import torch
import matplotlib.pyplot as plt

from torch import nn, optim #神经网络模型和优化器
from torch.utils.data import DataLoader, TensorDataset  #数据集和数据加载集

#1.准备数据
X = torch.randn(100, 1) #100行1列的随机数 
#预设真实值
w = torch.tensor([[2.0]]) #真实权重
b = torch.tensor([1.0])   #真实偏置
#定义随机噪声
noise = 0.1 * torch.randn(100, 1) 
#定义拟合的目标值
y = X @ w + b + noise 
#构建DATASET
dataset = TensorDataset(X, y)
#构建dataloader
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)  #指定数据源、批处理大小、打乱数据顺序

#2.构建模型
model = nn.Linear(in_features=1, out_features=1) #线性回归模型   一元

#3.定义损失函数和优化器
loss = nn.MSELoss()  #均方误差损失函数
optimizer = optim.SGD(model.parameters(), lr=0.001) #随机梯度下降优化器，学习率0.01

loss_avg = []  #存储每轮平均损失值

#4.模型训练
epoch_num = 1000  #训练轮数
for epoch in range(epoch_num):
    total_loss = 0  #每轮总损失值
    iter_num = 0 #每轮迭代次数
    # 一个轮次遍历Loader
    for x_train, y_train in dataloader:
        #4.1 前向传播
        y_pred = model(x_train)  #模型预测值 
        #4.2 计算损失
        loss_value = loss(y_pred, y_train) 
        total_loss += loss_value.item()  #累计损失值
        iter_num += 1
        #4.3 反向传播和优化 
        loss_value.backward()
        #4.4 更新参数
        optimizer.step()  #更新参数
        #4.5 清零梯度
        optimizer.zero_grad() #清零梯度

    #计算本轮平均损失
    loss_avg.append(total_loss / iter_num)

#打印参数
print(model.weight)
print(model.bias)

#画图
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
# 1 训练损失随轮次epoch变化
ax[0].plot(loss_avg)
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
# 2 绘制散点图和拟合直线
ax[1].scatter(X, y)
y_pred = model.weight.item() * X + model.bias.item()
ax[1].plot(X, y_pred, color='red', label='Fitted Line')
plt.show()
        