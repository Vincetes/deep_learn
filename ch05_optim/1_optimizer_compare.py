import sys
import os

# 获取项目根目录（即 E:\DeepLearn）
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将根目录加入搜索路径
sys.path.append(root_dir)  #导入根目录以后，就可以了


import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict # 有序字典
from common.optimizer import *

# 定义目标函数：f(x, y) = 1/20 x^2 + y^2
def f(x, y):
    return x ** 2 / 20 + y ** 2

# 定义梯度计算方法，得到一个长度为2的向量
def f_grad(x, y):
    return x / 10, 2 * y

# 定义初始点位置
init_pos = (-7.0, 2.0)

# 定义参数和梯度
params = {}
grads = {}

# 定义优化器，指定学习率
optimizers = OrderedDict()

# 优化后参数：兼顾收敛速度、稳定性，30次迭代内可清晰看到优化轨迹
optimizers['SGD'] = SGD(lr=0.4)  # 降低学习率，避免震荡
optimizers['Momentum'] = Momentum(lr=0.15, momentum=0.85)  # 小幅提lr，加速收敛
optimizers['AdaGrad'] = AdaGrad(lr=1.5)  # 保持不变，自适应场景适配
optimizers['Adam'] = Adam(lr=0.1, alpha1=0.9, alpha2=0.999)  # 修正一阶动量，补充二阶动量

idx = 1 # 子图序号

# 遍历优化器，用优化器更新参数求解最小值点
for key in optimizers:
    optimizer = optimizers[key]
    # 记录参数点更新的历史
    x_history = []
    y_history = []
    # 参数初始化
    params['x'], params['y'] = init_pos[0], init_pos[1]
    # 指定迭代 30 次
    for i in range(15):
        # 先保存当前点坐标
        x_history.append(params['x'])
        y_history.append(params['y'])
        # 1. 计算梯度
        grads['x'], grads['y'] = f_grad(params['x'], params['y'])
        # 2. 更新参数
        optimizer.update(params, grads)

    # 画图
    x = np.arange(-10, 10, 0.01)
    y = np.arange(-5, 5, 0.01)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    Z[Z > 7] = 0    # Z高度大于7的部分不画
    # 定义子图
    plt.subplot(2, 2, idx)
    idx += 1
    # 绘制等高线
    plt.contour(X, Y, Z)
    # 单独画出最小值点
    plt.plot(0, 0, '+')
    # 画出点轨迹曲线
    plt.plot(x_history, y_history, 'o-', color='red', markersize=2, label=key)
    plt.xlim(-10, 10)
    plt.ylim(-5, 5)
    plt.legend(loc='best')

plt.show()