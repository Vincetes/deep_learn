import numpy as np
import matplotlib.pyplot as plt

from common.gradient import numerical_gradient

#定义梯度下降法的函数
def gradient_descent(f, init_x, lr=0.01, num_iter=100):
    x = init_x #初始化参数
    x_history = [] #保存每一步的参数
    #循环迭代
    for i in range(num_iter):
        x_history.append(x.copy())
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x, np.array(x_history)

#定义目标函数f1(x1, x2) = x1^2 + x2^2
def f( x):
    return x[0]**2 + x[1]**2

if __name__ == '__main__':
    #1.定义初始值
    init_x = np.array([-3.0, 4.0])
    #2.定义超参数
    lr = 0.7
    num_iter = 100
    #3.使用梯度下降算法
    x, x_history = gradient_descent(f, init_x, lr, num_iter)
    print("最小值点为：", x)

    #绘制结果
    plt.plot([-5, 5], [0, 0], '--b')
    plt.plot([0, 0], [-5, 5], '--b')
    plt.scatter(x_history[:, 0], x_history[:, 1])
    plt.xlim([-3.5, 3.5])
    plt.ylim([-4.5, 4.5])
    plt.xlabel("X[0]")
    plt.ylabel("X[1]")

    plt.show()