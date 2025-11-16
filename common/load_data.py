import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def get_data():
    # 1. 从指定路径读取CSV格式的训练数据集
    # data是DataFrame格式（表格数据），包含label列（标签）和pixel0-pixel783列（特征）
    data = pd.read_csv('../data/train.csv')

    # 2. 划分数据集
    X = data.drop('label', axis=1)
    y = data['label']
    #固定的随机种子，可以让结果重现
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 3. 特征工程：归一化
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    #4.将数据转化为ndarray
    y_train = y_train.values
    y_test = y_test.values

    return x_train, x_test, y_train, y_test
