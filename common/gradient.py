import numpy as np

# 数值微分求导，传入x是一个标量
def numerical_diff(f, x):  #简单表达出来即可
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)  #中心差分 误差更小


# 数值微分求梯度，传入x是一个向量 一维
def _numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)  #创建一个和x形状相同的向量，用于保存梯度结果
    #遍历x中的特征xi
    for i in range(x.shape[0]):
        tmp = x[i]
        x[i] = tmp + h
        fxh1 = f(x)
        x[i] = tmp - h
        fxh2 = f(x)
        #利用中心差分公式计算偏导数
        grad[i] = (fxh1 - fxh2) / (2*h)
        #恢复x[i]的值
        x[i] = tmp
    return grad

#传入的X是一个矩阵
def numerical_gradient(f, X):
    #判断维度
    if X.ndim == 1:
        return _numerical_gradient(f, X)
    else:
        grad = np.zeros_like(X)
        #遍历矩阵中的每一行计算梯队
        for i, x in enumerate(X):
            grad[i] = _numerical_gradient(f, x)
    return grad
