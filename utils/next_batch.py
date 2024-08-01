import math


def next_batch_gt_5view(X1, X2, X3, X4, X5, gt, batch_size):
    # generate next batch for 5 view data with label
    tot = X1.shape[0]
    total = math.ceil(tot / batch_size)  # fix the last batch
    for i in range(int(total) - 1):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)
        batch_x1 = X1[start_idx: end_idx, ...]
        batch_x2 = X2[start_idx: end_idx, ...]
        batch_x3 = X3[start_idx: end_idx, ...]
        batch_x4 = X4[start_idx: end_idx, ...]
        batch_x5 = X5[start_idx: end_idx, ...]
        gt_now = gt[start_idx: end_idx, ...]
        yield batch_x1, batch_x2, batch_x3, batch_x4, batch_x5, gt_now, (i + 1)


"""
这段代码定义了一个生成器函数next_batch_gt_3view,用于生成具有标签的三个视图数据的下一个批次。
函数的参数包括：
X1、X2、X3:三个视图的数据。
gt:标签数据。
batch_size:批次大小。
函数的主要逻辑是根据给定的数据和批次大小，生成下一个批次的数据。具体步骤如下：
获取数据的总样本数,存储在变量tot中。
根据样本数和批次大小计算出总批次数,使用math.ceil函数向上取整,确保最后一个批次大小可以不足批次大小。
使用循环迭代每个批次,从start_idx到end_idx选择对应的数据。
将选择的数据存储在batch_x1、batch_x2、batch_x3和gt_now变量中。
使用yield关键字返回一个元组,包含batch_x1、batch_x2、batch_x3、gt_now以及当前批次的索引(i + 1)。
这个生成器函数可以在循环中使用，用于批量地获取下一个批次的数据和对应的标签。通过使用生成器函数，可以提高内存的利用效率，逐批次地加载数据进行训练。
"""
