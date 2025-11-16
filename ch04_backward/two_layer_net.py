
#softmax用于多类别分类问题  sigmoid函数用于二分类问题  croos_entropy用于多类别分类问题（损失函数
from common.gradient import numerical_gradient

#反向传播实现一下
from collections import OrderedDict #有序字典，用来保存层结构
from common.layers import *


#两层神经网络
class TwoLayerNet():
    #初始化
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size) #这是全部设置为0
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        #定义层结构
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        #单独定义最后一层
        self.lastLayer = SoftmaxWithLoss()

    #前向传播
    def forward(self, x):
        #对于神经网络中的每一层，一次调用forward()方法
        for layer in self.layers.values():
            y = layer.forward(x)
            x = y
        return x

    #计算损失
    def loss(self, x, t):
        y = self.forward(x)
        loss_value = self.lastLayer.forward(y, t)
        return loss_value

    #计算准确率
    def accuracy(self, x, t):
        y_pred = self.forward(x)
        y = np.argmax(y_pred, axis=1)
        accuracy = np.sum(y == t) / x.shape[0]  #直接相除计算百分比
        return accuracy

    #计算梯度  使用数值微分记录
    def numerical_gradient(self, x, t):
        #定义目标函数  目标函数就是损失函数
        loss_w = lambda _: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_w, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_w, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_w, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_w, self.params['b2'])
        return grads

    #使用反向传播
    def gradient(self, x, t):
        #前向传播，计算损失
        self.loss(x, t)
        #反向传播
        dy = 1
        dy = self.lastLayer.backward(dy)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dy = layer.backward(dy)
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        return grads

