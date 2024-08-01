import numpy as np


def normalize(x):
    """ Normalize 标准化"""
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x


def cal_classify(logger, *arg):
    """ print classification results打印分类结果 """
    if len(arg) == 3:
        logger.info(arg[0])
        logger.info(arg[1])
        logger.info(arg[2])
        output = """ 
                     ACC {:.2f} std {:.2f}
                     Precision {:.2f} std {:.2f} 
                     F-measure {:.2f} std {:.2f}""".format(np.mean(arg[0]) * 100, np.std(arg[0]) * 100,
                                                           np.mean(arg[1]) * 100,
                                                           np.std(arg[1]) * 100, np.mean(arg[2]) * 100,
                                                           np.std(arg[2]) * 100)
        logger.info(output)
        output2 = str(round(np.mean(arg[0]) * 100, 2)) + ',' + str(round(np.std(arg[0]) * 100, 2)) + ';' + \
                  str(round(np.mean(arg[1]) * 100, 2)) + ',' + str(round(np.std(arg[1]) * 100, 2)) + ';' + \
                  str(round(np.mean(arg[2]) * 100, 2)) + ',' + str(round(np.std(arg[2]) * 100, 2)) + ';'
        logger.info(output2)
        return round(np.mean(arg[0]) * 100, 2), round(np.mean(arg[1]) * 100, 2), round(np.mean(arg[2]) * 100, 2)
    elif len(arg) == 1:
        logger.info(arg)
        output = """ACC {:.2f} std {:.2f}""".format(np.mean(arg) * 100, np.std(arg) * 100)
        logger.info(output)
    return
# 该代码定义了一个名为cal_classify的函数，用于打印分类结果的统计信息。
# 函数的参数包括一个logger对象和可变数量的位置参数*arg。logger对象用于记录日志信息。位置参数*arg可以接受任意数量的参数。
# 函数首先通过len(arg)判断传入的参数数量。如果arg的长度为3，则表示传入了三个参数，按照顺序分别为准确率（ACC）、精确率（Precision）和F值（F-measure）。
# 然后，函数会通过logger.info()方法将每个参数的值打印到日志中。
# 接下来，函数使用np.mean()和np.std()函数分别计算参数的平均值和标准差，并格式化成字符串。这些统计信息包括ACC的平均值和标准差、Precision的平均值和标准差、F-measure的平均值和标准差。
# 这些统计信息会通过logger.info()方法打印到日志中。
# 接着，函数将统计信息组装成一个字符串output，并通过logger.info()方法打印到日志中。
# 然后，函数将统计信息组装成另一个字符串output2，并通过logger.info()方法打印到日志中。
# 最后，函数返回ACC、Precision和F-measure的平均值，分别通过np.mean()计算后乘以100，并使用round()函数保留两位小数。
# 如果传入的参数数量为1，则表示只传入了ACC参数。函数会将ACC的平均值和标准差计算并格式化成字符串，然后通过logger.info()方法打印到日志中。
# 整体上，该函数用于打印分类结果的统计信息，包括ACC、Precision和F-measure的平均值和标准差。
