# 导入必要的库
import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
import pandas as pd  # 数据处理库
import matplotlib.pyplot as plt  # 数据可视化库
from sklearn.model_selection import train_test_split  # 划分训练集和测试集
from sklearn.compose import ColumnTransformer  # 列转换器，用于同时处理不同类型特征
from sklearn.pipeline import Pipeline  # 管道操作，将多个处理步骤串联
from sklearn.impute import SimpleImputer  # 缺失值处理工具
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # 标准化和独热编码工具
from torch.utils.data import TensorDataset, DataLoader  # 构建数据集和数据加载器

# 创建数据集函数
def create_dataset():
    # 1. 从CSV文件读取房价数据
    data = pd.read_csv('./data/house_prices.csv') #vscode中当前工作文件夹是项目根目录
    # 2. 去除无关列"Id"（对预测房价无帮助）
    data.drop(["Id"], axis=1, inplace=True)
    # 3. 划分特征(X)和目标变量(y)，目标变量为房价(SalePrice)
    X = data.drop("SalePrice", axis=1)  #不修改原数据
    y = data["SalePrice"]
    # 4. 划分训练集和测试集，测试集占比20%，随机种子固定保证结果可复现
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22) #随机种子保证划分数据一致
    
    # 5. 特征工程（特征转换）
    # 5.1 按数据类型划分数值型和类别型特征
    numerical_features = X.select_dtypes(exclude=['object']).columns  # 数值型特征（排除object类型）
    categorical_features = X.select_dtypes(include=['object']).columns  # 类别型特征（仅包含object类型）
    
    # 5.2 定义数值型特征处理管道
    numerical_transformer = Pipeline(
        steps=[
            ('fillna', SimpleImputer(strategy='mean')),  # 用平均值填充缺失值
            ('std', StandardScaler())  # 标准化处理（均值为0，方差为1）
        ]
    )
    
    # 5.3 定义类别型特征处理管道
    categorical_transformer = Pipeline(
        steps=[
            ('fillna', SimpleImputer(strategy='constant', fill_value='NaN')),  # 用'NaN'填充缺失的类别值
            ('onehot', OneHotEncoder(handle_unknown='ignore'))  # 独热编码，忽略未知类别
        ]
    )
    
    # 5.4 组合列转换器，分别处理数值型和类别型特征
    transformer = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),  # 应用于数值型特征
            ('cat', categorical_transformer, categorical_features)  # 应用于类别型特征
        ]
    )
    
    # 5.5 对训练集和测试集进行特征转换   这里才调用
    x_train = transformer.fit_transform(x_train)  # 训练集：拟合并转换
    x_test = transformer.transform(x_test)  # 测试集：仅转换（使用训练集的拟合参数）
    
    # 将转换后的数组转换为DataFrame，保留特征名称
    x_train = pd.DataFrame(x_train.toarray(), columns=transformer.get_feature_names_out()) #获取特征名是内置函数
    x_test = pd.DataFrame(x_test.toarray(), columns=transformer.get_feature_names_out())
    
    # 6. 构建PyTorch张量数据集（特征和标签对应）
    train_dataset = TensorDataset(torch.tensor(x_train.values).float(), torch.tensor(y_train.values).float())
    test_dataset = TensorDataset(torch.tensor(x_test.values).float(), torch.tensor(y_test.values).float())
    
    # 返回训练集、测试集和特征数量
    return train_dataset, test_dataset, x_train.shape[1]



# 测试数据加载流程
# 1. 调用函数创建数据集
train_dataset, test_dataset, feature_num = create_dataset()
print(f"特征数量: {feature_num}")  # 打印特征数量


# 2. 定义神经网络模型
model = nn.Sequential(
    nn.Linear(feature_num, 128),  # 输入层：特征数量 -> 128维
    nn.BatchNorm1d(128),  # 批标准化，加速训练并稳定模型
    nn.ReLU(),  # ReLU激活函数，引入非线性
    nn.Dropout(0.2),  # Dropout层，随机丢弃20%神经元防止过拟合
    nn.Linear(128, 1),  # 输出层：128维 -> 1维（预测房价）
)


# 3. 自定义损失函数：对数均方根误差（Log RMSE）
def log_rmse(y_pred, target):
    # 将预测值限制在[1, +∞)，避免对数运算出错
    y_pred = torch.clamp(y_pred, 1, float("inf"))
    mse = nn.MSELoss()  # 均方误差损失
    # 计算预测值和目标值的对数MSE，再开平方
    return torch.sqrt(mse(torch.log(y_pred), torch.log(target)))

# 4. 模型训练和测试函数
def train_test(model, train_dataset, test_dataset, lr, epoch_num, batch_size, device):
    # 初始化模型参数的函数
    def init_params(layer):
        if isinstance(layer, nn.Linear):  # 对线性层进行参数初始化
            nn.init.xavier_normal_(layer.weight)  # 使用Xavier正态分布初始化权重
    
    # 1.1 初始化模型参数
    model.apply(init_params)
    # 1.2 将模型加载到指定设备（GPU或CPU）
    model = model.to(device)
    # 1.3 定义优化器（Adam优化器）
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 存储训练和测试损失的列表
    train_loss_list = []
    test_loss_list = []

    # 2. 开始训练循环
    for epoch in range(epoch_num):
        model.train()  # 切换到训练模式（启用Dropout等）
        # 2.1 创建训练数据加载器（按批次加载数据，打乱顺序）
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_loss_total = 0  # 累计训练损失
        
        # 2.2 按批次迭代训练
        for batch_idx, (X, y) in enumerate(train_loader):
            # 将数据加载到设备
            X, y = X.to(device), y.to(device)
            # 2.3.1 前向传播：计算预测值
            y_pred = model(X)
            # 2.3.2 计算损失（使用自定义的log_rmse）
            loss_value = log_rmse(y_pred.squeeze(), y)  # squeeze()去除多余维度
            # 2.3.3 反向传播：计算梯度
            loss_value.backward()
            # 2.3.4 更新参数
            optimizer.step()
            optimizer.zero_grad()  # 梯度清零，避免累积
            
            # 累加批次损失（乘以批次大小，最后求平均）
            train_loss_total += loss_value.item() * X.shape[0]

        # 计算本轮训练的平均损失
        this_train_loss = train_loss_total / len(train_dataset)
        train_loss_list.append(this_train_loss)

        # 3. 测试阶段
        model.eval()  # 切换到评估模式（关闭Dropout等）
        # 3.1 创建测试数据加载器
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        test_loss_total = 0  # 累计测试损失
        
        with torch.no_grad():  # 关闭梯度计算，节省内存和计算资源
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                loss_value = log_rmse(y_pred.squeeze(), y)
                test_loss_total += loss_value.item() * X.shape[0]
        
        # 计算本轮测试的平均损失
        this_test_loss = test_loss_total / len(test_dataset)
        test_loss_list.append(this_test_loss)

        # 打印本轮训练和测试损失
        print(f"epoch: {epoch+1}, train loss: {this_train_loss:.6f}, test loss: {this_test_loss:.6f}")

    # 返回训练和测试损失列表
    return train_loss_list, test_loss_list

# 选择计算设备（优先使用GPU，否则使用CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义超参数
lr = 0.1  # 学习率
epoch_num = 200  # 训练轮数
batch_size = 64  # 批次大小

# 调用训练测试函数
train_loss_list, test_loss_list = train_test(model, train_dataset, test_dataset, lr, epoch_num, batch_size, device)

# 绘制训练和测试损失曲线
plt.plot(train_loss_list, 'r-', label='train loss', linewidth=3)  # 训练损失：红色实线
plt.plot(test_loss_list, 'k--', label='test loss', linewidth=2)  # 测试损失：黑色虚线
plt.legend()  # 显示图例
plt.show()  # 显示图像