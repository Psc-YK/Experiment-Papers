import os
import random
import sys
import numpy as np
import scipy.io as sio
from scipy import sparse
from sklearn.model_selection import train_test_split
from utils import util


def load_multiview_data(config):
    data_name = config['dataset']
    main_dir = sys.path[0]
    X_list = []
    Y_list = []
    # 这段代码定义了一个名为 load_multiview_data 的函数，它接受一个 config 参数作为输入。
    # 函数首先从 config 中获取 dataset 的值，并将其赋值给 data_name 变量。
    # 然后，它使用 sys.path[0] 获取当前脚本所在的目录，并将其赋值给 main_dir 变量。这个目录将作为数据的主目录。即该所有文件所在的文件夹下作为主目录！
    # 接下来，函数创建了两个空列表 X_list 和 Y_list，用于存储多视图数据和标签数据。
    # 这段代码的目的是准备加载多视图数据的环境，包括设置数据集名称和获取当前脚本所在的目录，并创建空列表来存储数据和标签。
    if data_name in ['Scene_15']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', 'Scene_15.mat'))
        X = mat['X'][0]
        X_list.append(X[0].astype('float32'))  # 20 
        X_list.append(X[1].astype('float32'))  # 59
        X_list.append(X[2].astype('float32'))  # 40
        Y_list.append(np.squeeze(mat['Y']))
        Y_list.append(np.squeeze(mat['Y']))
    # 这段代码根据 data_name 的值加载特定数据集的多视图数据和标签数据。
    # 如果 data_name 的值是 'Scene_15'，则执行以下步骤：
    # 首先，使用 os.path.join(main_dir, 'data', 'Scene_15.mat') 构建数据文件的完整路径，并使用 sio.loadmat 函数加载 .mat 文件。
    # 然后，从加载的 .mat 文件中获取键为 'X' 和 'Y' 的数据。这些键分别对应多视图数据和标签数据。
    # 接下来，将多视图数据分别添加到 X_list 列表中。X[0]、X[1] 和 X[2] 分别表示多视图数据的不同视图。使用 .astype('float32') 将数据的数据类型转换为 float32。
    # 然后，将标签数据添加到 Y_list 列表中。np.squeeze(mat['Y']) 用于将标签数据的维度降为一维。
    # 这段代码的目的是根据特定的数据集名称，加载对应的多视图数据和标签数据，并将它们添加到 X_list 和 Y_list 列表中。
    elif data_name in ['LandUse_21']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', 'LandUse_21.mat'))
        train_x = []
        train_x.append(sparse.csr_matrix(mat['X'][0, 0]).A)  # 20
        train_x.append(sparse.csr_matrix(mat['X'][0, 1]).A)  # 59
        train_x.append(sparse.csr_matrix(mat['X'][0, 2]).A)  # 40
        index = random.sample(range(train_x[0].shape[0]), 2100)  # 第一个视图中有2100行数据
        for view in [1, 2, 0]:  # 表示依次取值1、2、0
            x = train_x[view][index]  # 分别将三个视图的数据记录到X中
            y = np.squeeze(mat['Y']).astype('int')[index]
            X_list.append(x)  # 将三个视图的数据合为一个数据
            Y_list.append(y)
    # 这段代码是针对数据集名称为 'LandUse_21' 的情况。它加载了名为 'LandUse_21.mat' 的 .mat 数据文件，并进行了处理。
    # 首先，使用 os.path.join(main_dir, 'data', 'LandUse_21.mat') 构建数据文件的完整路径，并使用 sio.loadmat 函数加载 .mat 文件。
    # 然后，将加载的 .mat 文件中的数据存储在 train_x 列表中。train_x 列表包含三个视图的稀疏矩阵数据。
    # 通过使用 sparse.csr_matrix(mat['X'][0, view]).A 将稀疏矩阵转换为稠密数组。
    # 接下来，使用 random.sample 函数从第一个视图的数据中随机选择 2100 个样本索引，并将中的样本索引存储在 index 列表中。
    # 然后，使用循环遍历 [1, 2, 0]，对每个视图进行以下操作：
    #
    # 从 train_x 中根据视图索引获取相应的数据，并使用选中的样本索引 index 进行筛选，得到 x 变量。
    # 从 .mat 文件中获取标签数据 mat['Y']，并将其转换为整数类型，并使用选中的样本索引 index 进行筛选，得到 y 变量。
    # 将 x 添加到 X_list 列表中。
    # 将 y 添加到 Y_list 列表中。
    #
    # 这段代码的目的是加载 'LandUse_21' 数据集的多视图数据和标签数据，并根据选中的样本索引将它们添加到 X_list 和 Y_list 列表中。

    return X_list, Y_list
