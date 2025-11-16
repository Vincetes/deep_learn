# 导入依赖库：数据处理、模型加载、数据集划分、归一化、激活函数
import numpy as np  # 数值计算核心库（矩阵乘法、数组操作等）
import pandas as pd  # 数据读取与处理库（专门处理CSV等表格数据）
import joblib  # 模型保存/加载库（用于读取预训练的神经网络参数）
from sklearn.model_selection import train_test_split  # 数据集划分工具（拆分训练集/测试集）
from sklearn.preprocessing import MinMaxScaler  # 特征归一化工具（缩放特征到[0,1]）
from common.functions import sigmoid, softmax  # 导入自定义激活函数（sigmoid用于隐藏层，softmax用于输出层）

# 定义函数：获取并预处理测试数据（该函数暂时只返回测试集，用于模型测试）
def get_data():
    # 1. 从指定路径读取CSV格式的训练数据集
    # data是DataFrame格式（表格数据），包含label列（标签）和pixel0-pixel783列（特征）
    data = pd.read_csv('../data/train.csv')

    # 2. 拆分特征（X）和标签（y），并划分训练集/测试集
    X = data.drop('label', axis=1)  # 特征矩阵X：删除label列，保留所有pixel列（shape: [样本数, 784]）
    y = data['label']  # 标签向量y：提取label列（shape: [样本数,]，值为0-9的整数）
    # 划分数据集：test_size=0.2表示测试集占20%，训练集占80%；random_state=42固定随机种子（保证划分结果可复现）
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 注：代码中暂时不用训练集（x_train/y_train），仅返回测试集用于评估预训练模型

    # 3. 特征工程：归一化（消除特征量纲差异，避免模型偏向数值大的特征）
    scaler = MinMaxScaler()  # 创建归一化工具实例（默认缩放范围[0,1]）
    x_train = scaler.fit_transform(x_train)  # 对训练集：先拟合（计算x_train的min/max）再转换（缩放至[0,1]）
    x_test = scaler.transform(x_test)  # 对测试集：仅用训练集的min/max转换（避免测试集数据泄露）

    return x_test, y_test  # 返回预处理后的测试集特征和真实标签（用于模型测试）

# 定义函数：初始化神经网络（加载预训练的模型参数）
def init_network():
    # 从指定路径加载预训练的神经网络参数（joblib保存的字典格式）
    # 字典中包含：W1/W2/W3（各层权重矩阵）、b1/b2/b3（各层偏置向量）
    network = joblib.load('../data/nn_sample')
    return network  # 返回加载好的模型参数（供前向传播使用）

# 定义函数：前向传播（核心！输入模型参数和特征，输出预测概率分布）
def forward(network, x):
    # 从模型参数字典中提取各层权重（W）和偏置（b）
    w1, w2, w3 = network['W1'], network['W2'], network['W3']  # W1: [784, 隐藏层1神经元数], W2: [隐藏层1, 隐藏层2], W3: [隐藏层2, 10]
    b1, b2, b3 = network['b1'], network['b2'], network['b3']  # b1: [隐藏层1神经元数,], b2: [隐藏层2神经元数,], b3: [10,]

    # 第1隐藏层计算：线性变换 + 非线性激活（sigmoid）
    a1 = np.dot(x, w1) + b1  # 线性变换：输入x × 权重W1 + 偏置b1（shape: [样本数, 隐藏层1神经元数]）
    z1 = sigmoid(a1)  # 非线性激活：sigmoid函数将输出映射到(0,1)，引入模型非线性表达能力

    # 第2隐藏层计算：与第1隐藏层逻辑一致
    a2 = np.dot(z1, w2) + b2  # 线性变换：第1隐藏层输出z1 × 权重W2 + 偏置b2（shape: [样本数, 隐藏层2神经元数]）
    z2 = sigmoid(a2)  # 非线性激活：进一步增强模型对复杂特征的拟合能力

    # 输出层计算：线性变换 + 激活函数（sigmoid→softmax）
    a3 = np.dot(z2, w3) + b3  # 线性变换：第2隐藏层输出z2 × 权重W3 + 偏置b3（shape: [样本数, 10]，10对应10个类别）
    z3 = sigmoid(a3)  # 中间激活（可选，实际中可直接用a3做softmax，此处是设计选择）
    y = softmax(z3)  # 输出层激活：将10个神经元的原始输出转换为概率分布（每行和=1，shape: [样本数, 10]）

    return y  # 返回每个样本的10个类别概率分布

# 主流程：模型测试与准确率评估（核心执行逻辑）
# 1. 获取预处理后的测试集数据
x, y = get_data()  # x: 测试集特征（shape: (8400, 784)）；y: 测试集真实标签（shape: (8400,)）
# print(x.shape)  # 可选打印：验证x的形状（8400个样本，每个样本784个特征）
# print(y.shape)  # 可选打印：验证y的形状（8400个样本的真实标签）

# 2. 加载预训练的神经网络模型参数
network = init_network()  # network是包含W1/W2/W3/b1/b2/b3的字典
print("W1:", network['W1'].shape)
print("W2:", network['W2'].shape)
print("W2:", network['W2'].shape)
print("W3:", network['W3'].shape)
print("b1:", network['b1'].shape)
print("b2:", network['b2'].shape)
print("b3:", network['b3'].shape)
print("b3:", network['b3'].shape)

# 3. 执行前向传播，得到测试集的预测概率分布
y_proba = forward(network, x)  # y_proba: 预测概率（shape: (8400, 10)，每行是一个样本的10类概率）
print(y_proba.shape)  # 打印概率分布形状，验证输出是否符合预期（8400个样本×10个类别）

# 4. 将概率分布转换为最终分类标签（从“概率”到“明确类别”）
y_pred = np.argmax(y_proba, axis=1)  # axis=1：按行取最大值的索引（索引0-9对应类别0-9），shape: (8400,)
# 示例：若某样本y_proba行是[0.01, 0.85, ..., 0.02]，最大值索引1→预测类别1

# 5. 计算模型分类准确率（评估模型性能）
accuracy_cnt = np.sum(y == y_pred)  # 统计预测标签与真实标签一致的样本数（y和y_pred都是(8400,)，逐元素对比后求和）
n = x.shape[0]  # 获取测试集总样本数（x.shape[0]是行数，即8400）
print("Accuracy:", accuracy_cnt / n)  # 准确率=正确样本数/总样本数，输出格式如：Accuracy: 0.95（95%准确率）
