import numpy as np

# 随机梯度下降 SGD
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
    #参数更新 传入参数字典和梯度字典
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


# 动量法 Momentum 迭代效果更高
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None #v是相应的历史

    # 参数更新方法
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]

#自适应梯度 AdaGrad
class AdaGrad:
    def __init__(self, lr=0.01, eps=1e-7):
        self.lr = lr
        self.h = None  #历史梯度的平方和
        self.eps = eps

    #参数更新方法
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        #按照公式进行参数更新
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + self.eps)

#均方根传播
class RMSprop:
    def __init__(self, lr=0.01, decay=0.9, eps=1e-7):
        self.lr = lr
        self.decay = decay
        self.h = None
        self.eps = eps

    #参数更新方法
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        #按照公式
        for key in params.keys():
            self.h[key] = self.decay * self.h[key] + (1 - self.decay) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + self.eps)

# Adam  自适应矩估计
class Adam:
    # 初始化
    def __init__(self, lr=0.01, alpha1=0.9, alpha2=0.999):
        self.lr = lr
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.v = None
        self.h = None
        self.t = 0  # 迭代次数
    # 更新方法
    def update(self, params, grads):
        # 初始化v和h
        if self.v is None:
            self.v, self.h = {}, {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
                self.h[key] = np.zeros_like(val)
        self.t += 1 # 迭代次数+1
        # 按照当前的迭代次数，改变学习率参数  这个地方其实是不变的
        lr_t = self.lr * np.sqrt(1 - self.alpha2**self.t) / (1 - self.alpha1**self.t)
        # 遍历所有参数，按公式进行更新
        for key in params.keys():
            # self.v[key] = self.alpha1 * self.v[key] + (1 - self.alpha1) * grads[key]
            # self.h[key] = self.alpha2 * self.h[key] + (1 - self.alpha2) * (grads[key] ** 2)
            self.v[key] += (1 - self.alpha1) * (grads[key] - self.v[key])
            self.h[key] += (1 - self.alpha2) * (grads[key] ** 2 - self.h[key])
            params[key] -= lr_t * self.v[key] / (np.sqrt(self.h[key]) + 1e-8)