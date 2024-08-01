import numpy as np
from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder

"""
该代码定义了一个名为get_mask的函数，用于生成缺失数据的掩码矩阵。
缺失数据设为0
"""


def get_mask(view_num, alldata_len, missing_rate):
    """Randomly generate incomplete data information, simulate partial view data with complete view data
    :param view_num:view number
    :param alldata_len:number of samples
    :param missing_rate:Defined in section 4.1 of the paper
    :return: mask
    """
    #  view_num：视图的数量，即数据有多少个不同的视角。
    #  alldata_len：样本的总数量。
    #  missing_rate：缺失率，表示数据中有多少比例是缺失的。
    # 该函数的主要目的是生成一个掩码矩阵，用于标记哪些数据是缺失的，哪些是可用的。矩阵的形状是 (样本数量, 视图数量)，
    # 矩阵中的每个元素是1或0，1表示对应的视图数据存在，0表示数据缺失。

    full_matrix = np.ones((int(alldata_len * (1 - missing_rate)), view_num))  # 生成矩阵,若missing_rate为0，则生成的是全部为1的矩阵

    alldata_len = alldata_len - int(alldata_len * (1 - missing_rate))
    missing_rate = 0.5
    if alldata_len != 0:
        one_rate = 1.0 - missing_rate
        if one_rate <= (1 / view_num):
            enc = OneHotEncoder()  # n_values=view_num
            view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
            full_matrix = np.concatenate([view_preserve, full_matrix], axis=0)
            choice = np.random.choice(full_matrix.shape[0], size=full_matrix.shape[0], replace=False)
            matrix = full_matrix[choice]
            return matrix
        error = 1
        if one_rate == 1:
            matrix = randint(1, 2, size=(alldata_len, view_num))
            full_matrix = np.concatenate([matrix, full_matrix], axis=0)
            choice = np.random.choice(full_matrix.shape[0], size=full_matrix.shape[0], replace=False)
            matrix = full_matrix[choice]
            return matrix
        while error >= 0.005:
            enc = OneHotEncoder()  # n_values=view_num
            view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
            one_num = view_num * alldata_len * one_rate - alldata_len
            ratio = one_num / (view_num * alldata_len)
            matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int)
            a = np.sum(((matrix_iter + view_preserve) > 1).astype(np.int))
            one_num_iter = one_num / (1 - a / one_num)
            ratio = one_num_iter / (view_num * alldata_len)
            matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int)
            matrix = ((matrix_iter + view_preserve) > 0).astype(np.int)
            ratio = np.sum(matrix) / (view_num * alldata_len)
            error = abs(one_rate - ratio)
        full_matrix = np.concatenate([matrix, full_matrix], axis=0)

    choice = np.random.choice(full_matrix.shape[0], size=full_matrix.shape[0], replace=False)
    matrix = full_matrix[choice]
    return matrix
# 该代码定义了一个名为get_mask的函数，用于生成缺失数据的掩码矩阵。
# 函数的参数包括view_num（视图数量）、alldata_len（样本数）和missing_rate（缺失率）。函数通过随机生成缺失数据的方式，模拟完整视图数据中的部分缺失。
# 首先，函数创建一个大小为alldata_len * (1 - missing_rate)的全1矩阵full_matrix，表示完整的视图数据。
# 然后，根据alldata_len的值判断是否需要生成缺失数据。如果alldata_len不为0，则表示需要生成缺失数据。
# 接下来，函数根据缺失率missing_rate计算出一个比例one_rate，表示完整数据中1的比例。
# 如果one_rate小于等于1 / view_num，则表示可以使用独热编码（OneHotEncoder）生成缺失数据。
# 函数使用randint()函数生成一个大小为alldata_len * view_num的随机整数矩阵作为缺失数据，并与full_matrix进行合并和随机重排，最后返回合并后的矩阵。
# 如果one_rate等于1，则表示所有数据都是1，函数使用randint()函数生成一个大小为alldata_len * view_num的矩阵，所有元素都是1或2，作为缺失数据，
# 并与full_matrix进行合并和随机重排，最后返回合并后的矩阵。
# 如果以上条件都不满足，则进入循环，根据缺失率和完整数据生成部分缺失数据。函数使用randint()函数生成一个大小为alldata_len * view_num的随机整数矩阵matrix_iter，
# 并根据缺失率和完整数据生成的view_preserve进行相加。然后计算新矩阵中大于1的元素个数，并计算出新的缺失率。循环直到新的缺失率与目标缺失率的误差小于0.005。
# 最后，将生成的缺失数据与完整数据进行合并和随机重排，最后返回合并后的矩阵。
# 总体上，该函数用于生成缺失数据的掩码矩阵，通过模拟随机生成缺失数据的方式来实现部分视图数据的缺失。
