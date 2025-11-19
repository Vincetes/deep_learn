import torch
import torch.nn as nn

#自定义神经网络类
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        #定义三个线性层
        self.linear1 = nn.Linear(3, 4)  #输入层到隐藏层
        nn.init.xavier_normal_(self.linear1.weight)  #使用Xavier初始化权重
        self.linear2 = nn.Linear(4, 4)  #隐藏层到隐藏层
        nn.init.kaiming_normal_(self.linear2.weight)  #使用Xavier初始化权重
        self.out = nn.Linear(4, 2)  #隐藏层到输出层

    #前向传播
    def forward(self, x):
        #定义前向传播
        x = self.linear1(x)
        x = torch.tanh(x)  #激活函数tanh
        x = self.linear2(x)
        x = torch.relu(x)  #激活函数ReLU
        x = self.out(x) 
        x = torch.softmax(x, dim=1)  #输出层使用Softmax激活函数
        return x
    
#测试
#1.创建模型实例
x = torch.randn(10, 3)  #创建一个2行3列的随机输入张量

#2.创建模型对象
model = Model()

#3.前向传播
output = model(x)

print("输出结果：", output)


print(model.linear1.weight)
print(model.linear1.bias)
print(model.linear2.weight)
print(model.linear2.bias)
print(model.out.weight)
print(model.out.bias)

print("____________________________________________________________")

#2.调用模型的parameters()方法，查看所有可训练参数
for param in model.parameters():
    print(param)    


print("____________________________________________________________")

#3.调用模型的state_dict()方法，查看所有可训练参数
state_dict = model.state_dict()
for key, value in state_dict.items():
    print(f"{key}: {value}")

print("____________________________________________________________")

#4.查看模型的架构和参数数量
from torchsummary import summary

summary(model, input_size=(3,), batch_size=10, device='cpu')  #输入尺寸为3
