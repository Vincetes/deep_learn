# 导入依赖库
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from common.functions import sigmoid, softmax


def get_data():
    data = pd.read_csv('../data/train.csv')

    X = data.drop('label', axis=1)
    y = data['label']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_test, y_test


def init_network():
    network = joblib.load('../data/nn_sample')
    return network


def forward(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, w3) + b3
    z3 = sigmoid(a3)
    y = softmax(z3)

    return y


x, y = get_data()

network = init_network()


#定义变量  分批次计算节省内存啥的
batch_size = 100
accuracy_cnt = 0
n = x.shape[0]

#3.循环迭代：分批次做测试，前向传播，并累积预测准确个数
for i in range(0, n, batch_size):
    #3.1取出当前批次的数据
    x_batch = x[i:i+batch_size]
    #3.2前向传播
    y_batch = forward(network, x_batch)
    #3.3将输出分类概率转换为分类标签
    y_pred = np.argmax(y_batch, axis=1)
    #3.4累加准确个数
    accuracy_cnt += np.sum(y[i:i + batch_size] == y_pred)

#4.计算分类准确率
print("Accuracy:", accuracy_cnt / n)