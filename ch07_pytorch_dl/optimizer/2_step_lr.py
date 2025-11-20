import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import StepLR

#定义函数
def f(x):
    return 0.05 * x[0] ** 2 + x[1] ** 2


#主流程
if __name__ == "__main__":  # 规范写法：确保主流程只在直接运行脚本时执行

    #1.参数X初始化
    X = torch.tensor([-7, 2.0], requires_grad=True)

    #2. 定义超参数
    lr = 0.9
    num_iters = 500

    #3.定义优化器
    optimizer = torch.optim.SGD([X], lr = lr)


    #4. 定义学习衰减策略
    lr_scheduler = StepLR(optimizer, step_size = 20, gamma = 0.7)

        #首先拷贝当前X的值
    X_arr = X.detach().numpy().copy()
    lr_list = []
    for i in range(num_iters):
        #前向传播
        y = f(X)
        #反向传播
        y.backward()
        #更新参数
        optimizer.step()
        #梯度清零
        optimizer.zero_grad()

        #将更新之后的x保存到列表中
        X_arr = np.vstack([X_arr, X.detach().numpy()])
        lr_list.append(optimizer.param_groups[0]['lr'])

        #更新学习率
        lr_scheduler.step()

    # 设置Matplotlib绘图时的字体为“楷体（KaiTi）”，解决中文显示为方框/乱码的问题
    plt.rcParams['font.sans-serif'] = ['KaiTi']

    # 解决Matplotlib中负号“-”显示为方块的问题（开启后负号可正常显示）
    plt.rcParams['axes.unicode_minus'] = False    
    
    # ---- 2. 准备画布 ----
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    x1_grid, x2_grid = np.meshgrid(
        np.linspace(-7, 7, 100),   # 注意：原图 num=1o → 100
        np.linspace(-2, 2, 100)
    )

    y_grid = 0.05 * x1_grid**2 + x2_grid**2


    # ---- 3. 左图：等高线 + 梯度下降轨迹 ----
    ax[0].contour(x1_grid, x2_grid, y_grid, levels=30, colors='gray')
    # 假设已有梯度下降结果数组 X_arr，形状 (n_steps, 2)
    # ax[0].plot(X_arr[:, 0], X_arr[:, 1], 'r')   # 红色轨迹
    ax[0].set_title("梯度下降过程")

    # ---- 4. 右图：学习率衰减曲线 ----
    # 假设已有学习率序列 ir_list
    print(lr_list)
    ax[1].plot(lr_list, 'k')
    ax[1].set_title("学习率衰减曲线")

    plt.tight_layout()
    plt.show()