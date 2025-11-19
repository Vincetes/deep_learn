import torch
from torch import nn, optim

#定义模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(5, 3)  #简单的线性层 
        self.linear.weight.data = torch.tensor([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [0.1, 0.3, 0.5],
            [0.2, 0.4, 0.6]
            ]).T
            #初始化权重为0.5
        self.linear.bias.data = torch.tensor([0.1, 0.2, 0.3])  #初始化偏置为0.1

    def forward(self, x):
        return self.linear(x)
    
#主流程
#1.定义数据 
# 输入值 2X5
X = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0],
                  [5.0, 4.0, 3.0, 2.0, 1.0]],
                  dtype=torch.float)  #输入张量

#目标值 2X3
target = torch.tensor([[0.5, 1.0, 1.5],
                       [1.5, 1.0, 0.5]], dtype=torch.float)  #目标张量


#2.创建模型实例
model = Model()

#3.前向传播，预测输出
output = model(X)

#4.定义损失函数
loss = nn.MSELoss()  #均方误差损失函数
loss_value = loss(output, target) 

#5.反向传播，计算梯度
loss_value.backward()

#6.定义优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)  #使用随机梯度下降优化器

#7.更新参数
optimizer.step()
optimizer.zero_grad()  #清零梯度


#打印模型参数
for param in model.state_dict():
    print(param)
    print(model.state_dict()[param])