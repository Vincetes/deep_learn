import torch
import matplotlib.pyplot as plt

x = torch.linspace(-5, 5, 1000, requires_grad=True)
y = torch.relu(x)

#画图
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
#1.原函数图像
ax[0].plot(x.data, y.data, "purple")
ax[0].set_title("Relu Function")

ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
ax[0].spines['left'].set_position("zero")
ax[0].spines['bottom'].set_position("zero")

# 反向传播，计算x的梯度
y.sum().backward()

# 2.导函数图像
ax[1].plot(x.data, x.grad.data, "purple")
ax[1].set_title("Gradient of Tanh Function")

ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[1].spines['left'].set_position("zero")
ax[1].spines['bottom'].set_position("zero")


plt.show()
