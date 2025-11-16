import numpy as np

from ch02_nn_base.ch03_digit_recognizer_batch import accuracy_cnt
from common.functions import softmax, sigmoid, cross_entropy
#softmax用于多类别分类问题  sigmoid函数用于二分类问题  croos_entropy用于多类别分类问题（损失函数
from common.gradient import numerical_gradient

#两层神经网络
class TwoLayerNet():
    #初始化
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size) #这是全部设置为0
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    #前向传播
    def forward(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = x @ W1 + b1
        a2 = a1 @ W2 + b2
        y = softmax(a2)
        return y

    #计算损失
    def loss(self, x, t):
        y = self.forward(x)
        loss_value = cross_entropy(y, t)
        return loss_value

    #计算准确率
    def accuracy(self, x, t):
        y_proba = self.forward(x)
        y_pred = np.argmax(y_proba, axis=1)
        accuracy = np.sum(y_pred == t) / x.shape[0]  #直接相除计算百分比
        return accuracy

    #计算梯度
    def numerical_gradient(self, x, t):
        #定义目标函数  目标函数就是损失函数
        loss_w = lambda _: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_w, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_w, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_w, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_w, self.params['b2'])
        return grads