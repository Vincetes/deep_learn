import torch
import torch.nn as nn
from torchsummary import summary

#1.定义数据
x = torch.randn(10, 3)  #创建一个10行3列的随机输入张量

#2.构建模型
model = nn.Sequential(
    nn.Linear(3, 4),  #输入层到隐藏层
    nn.Tanh(),        #激活函数tanh
    nn.Linear(4, 4),  #隐藏层到隐藏层
    nn.ReLU(),        #激活函数ReLU
    nn.Linear(4, 2),  #隐藏层到输出层
    nn.Softmax(dim=1) #输出层使用Softmax激活函数
)

#定义一个参数初始化的函数
def init_params(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)  #使用Xavier初始化权重
        nn.init.constant_(layer.bias, 0.1)            #偏置初始化为0

# 3.参数初始化
model.apply(init_params)

#4.前向传播
output = model(x)
print("输出结果：", output)

# 5. 修正summary的调用
# 先确定设备（如果有GPU则用cuda，否则用cpu）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 将模型移动到目标设备（和summary的device一致）
model = model.to(device)


summary(model, input_size=(3,), batch_size=10, device=device.type)  #输入尺寸为3