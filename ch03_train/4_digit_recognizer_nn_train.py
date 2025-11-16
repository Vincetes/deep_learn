import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

from ch02_nn_base.ch03_digit_recognizer_batch import batch_size, y_batch
from two_layer_net import TwoLayerNet
from common.load_data import get_data

#1.加载数据
x_train, x_test, y_train, y_test = get_data()

#2.创建模型
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

#3.设置参数
learning_rate = 0.1
batch_size = 100
num_epochs = 5

train_size = x_train.shape[0]
iter_per_epoch = np.ceil(train_size / batch_size)  #向上取整
iters_num = int(num_epochs * iter_per_epoch)  #总的迭代次数

train_loss_list = []
train_acc_list = []
test_acc_list = []

#4.循环迭代
for i in range(iters_num):
    #4.1 随机选取批量数据
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = y_train[batch_mask]

    #4.2计算梯度
    grad = network.numerical_gradient(x_batch, t_batch)
    print("grad ======= ", i)

    #4.3更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):  #这玩意直接也能对b这种直接加上去的求解梯度真是牛的
        network.params[key] -= learning_rate * grad[key]

    #4.4计算并保存当前的训练损失
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    #4.5每完成一个epoch的实现迭代，计算并保存训练准确率和测试准确率
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, y_train)
        test_acc = network.accuracy(x_test, y_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

#5.画图
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.legend(loc='lower right')
plt.show()





















