from common.functions import *

#Relu
class Relu:
    def __init__(self):
        # 内部属性，记录哪些数据x<=0
        self.mask = None

    #前向传播
    def forward(self, x):
        self.mask = (x <= 0)  #这里是布尔值 存储的是索引？
        y = x.copy()
        #将x <= 0的值都赋为0
        y[self.mask] = 0
        return y

    #反向传播
    def backward(self, dy):
        dx = dy.copy()
        #将x <= 0的值都赋为0
        dx[self.mask] = 0
        return dx

#Sigmoid
class Sigmoid:
    def __init__(self):
        #定义内部属性，记录输出值y，用于反向传播时计算梯度
        self.y = None

    #前向传播
    def forward(self, x):
        y = sigmoid( x)
        self.y = y
        return y

    #反向传播  dy即上游传递过来的梯度
    def backward(self, dy):
        dx = dy * (1.0 - self.y) * self.y  #基于求导公式得到结果
        return dx

#Affine 仿射层
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        #对输入数据X保存起来，用于反向传播时计算梯度
        self.x = None
        #将权重和参数保存起来，用于反向传播时计算梯度
        self.dW = None
        self.db = None

    #输入数据可能是多维的，将其转化为二维
    def forward(self, x):
        self.original_x_shape = x.shape
        self.x = x.reshape(x.shape[0], -1)
        y = x @ self.W + self.b
        return y

    def backward(self, dy):  #暂时不管你采用什么损失函数   这个dy即是L对于y求导以后的结果
        dx = dy @ self.W.T
        dx = dx.reshape(*self.original_x_shape)
        self.dW = self.x.T @ dy
        self.db = np.sum(dy, axis=0)   #这里现在还是不是很懂
        return dx

#输出层
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy(self.y, self.t)
        return self.loss

    def backward(self, dy=1):
        n = self.t.shape[0]
        # 如果是独热编码，就直接代入公式  其实就是结果多样，但是每个样本只有一个正确结果
        if self.t.size == self.y.size:
            dx = self.y - self.t
        # 如果不是独热编码，就需要找到分类号对应的值
        else:
            dx = self.y.copy()
            dx[np.arange(n), self.t] -= 1
        return dx / n
