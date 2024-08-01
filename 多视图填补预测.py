import argparse
import itertools
import time
import torch
import pandas as pd
from model_multiview import DCPMultiView
from utils.util import cal_classify
from utils.logger import get_logger
from utils.datasets import *
from configure.configure_supervised_multiview import get_default_config
import collections
import warnings

warnings.simplefilter("ignore")

dataset = {2: "MRI"}
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default='2', help='dataset id')
parser.add_argument('--devices', type=str, default='0', help='gpu device ids')
parser.add_argument('--print_num', type=int, default='50', help='gap of print evaluations')
parser.add_argument('--test_time', type=int, default='1', help='number of test times')       # 测试次数
parser.add_argument('--missing_rate', type=float, default='0', help='missing rate')
args = parser.parse_args()
dataset = dataset[args.dataset]


def main():
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.devices)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    config = get_default_config(dataset)
    config['missing_rate'] = args.missing_rate
    config['print_num'] = args.print_num
    config['dataset'] = dataset
    logger, plt_name = get_logger(config)
    logger.info('Dataset:' + str(dataset))
    for (k, v) in config.items():
        if isinstance(v, dict):
            logger.info("%s={" % (k))
            for (g, z) in v.items():
                logger.info("          %s = %s" % (g, z))
        else:
            logger.info("%s = %s" % (k, v))

    data = pd.read_csv(r'C:\Users\psck\Desktop\空值设为0并保留2位小数.csv')
    train_x = data.iloc[:, 0:121].values
    train_y = data.iloc[:, 121].values
    # print(train_x)
    # print(train_y)

    x1_train_raw = train_x[:, 0:5]
    x2_train_raw = train_x[:, 5:21]
    x3_train_raw = train_x[:, 21:28]
    x4_train_raw = train_x[:, 28:31]
    x5_train_raw = train_x[:, 31:121]
    label_raw = train_y

    fold_acc, fold_precision, fold_f_measure = [], [], []
    for data_seed in range(1, args.test_time + 1):
        start = time.time()
        np.random.seed(data_seed)
        len1 = x1_train_raw.shape[1]
        len2 = x2_train_raw.shape[1] + x1_train_raw.shape[1]
        len3 = x3_train_raw.shape[1] + x2_train_raw.shape[1] + x1_train_raw.shape[1]
        len4 = x4_train_raw.shape[1] + x3_train_raw.shape[1] + x2_train_raw.shape[1] + x1_train_raw.shape[1]
        len5 = len4 + x5_train_raw.shape[1]
        # data2 = np.concatenate([x1_train_raw, x2_train_raw, x3_train_raw, x4_train_raw, x5_train_raw], axis=1)

        x_train, x_test, labels_train, labels_test = train_test_split(train_x, label_raw, test_size=0.2)  # 测试集的比例为20%
        x1_train = x_train[:, :len1]
        x2_train = x_train[:, len1:len2]
        x3_train = x_train[:, len2:len3]
        x4_train = x_train[:, len3:len4]
        x5_train = x_train[:, len4:len5]

        x1_test = x_test[:, :len1]
        x2_test = x_test[:, len1:len2]
        x3_test = x_test[:, len2:len3]
        x4_test = x_test[:, len3:len4]
        x5_test = x_test[:, len4:len5]

        x1_train = torch.from_numpy(x1_train).float().to(device)
        x2_train = torch.from_numpy(x2_train).float().to(device)
        x3_train = torch.from_numpy(x3_train).float().to(device)
        x4_train = torch.from_numpy(x4_train).float().to(device)
        x5_train = torch.from_numpy(x5_train).float().to(device)

        x1_test = torch.from_numpy(x1_test).float().to(device)
        x2_test = torch.from_numpy(x2_test).float().to(device)
        x3_test = torch.from_numpy(x3_test).float().to(device)
        x4_test = torch.from_numpy(x4_test).float().to(device)
        x5_test = torch.from_numpy(x5_test).float().to(device)

        labels_train = np.array(labels_train)
        labels_test = np.array(labels_test)

        accumulated_metrics = collections.defaultdict(list)

        DCP = DCPMultiView(config)
        optimizer = torch.optim.Adam(
            itertools.chain(DCP.autoencoder1.parameters(), DCP.autoencoder2.parameters(),
                            DCP.autoencoder3.parameters(), DCP.autoencoder4.parameters(), DCP.autoencoder5.parameters(),
                            DCP.a2b.parameters(), DCP.b2a.parameters(),
                            DCP.a2c.parameters(), DCP.c2a.parameters(),
                            DCP.b2c.parameters(), DCP.c2b.parameters(),
                            DCP.b2d.parameters(), DCP.c2d.parameters(),
                            DCP.b2e.parameters(), DCP.e2b.parameters(),
                            DCP.a2d.parameters(), DCP.d2a.parameters(),
                            DCP.a2e.parameters(), DCP.e2a.parameters(),
                            DCP.c2d.parameters(), DCP.d2c.parameters(),
                            DCP.c2e.parameters(), DCP.e2c.parameters(),
                            DCP.d2e.parameters(), DCP.e2d.parameters(),
                            ),
            lr=config['training']['lr'])

        logger.info(DCP.autoencoder1)
        logger.info(DCP.a2b)
        logger.info(optimizer)

        DCP.autoencoder1.to(device), DCP.autoencoder2.to(device), DCP.autoencoder3.to(device), DCP.autoencoder4.to(
            device), DCP.autoencoder5.to(device)
        DCP.a2b.to(device), DCP.b2a.to(device)
        DCP.b2c.to(device), DCP.c2b.to(device)
        DCP.a2c.to(device), DCP.c2a.to(device)
        DCP.b2d.to(device), DCP.c2d.to(device)
        DCP.b2e.to(device), DCP.e2b.to(device)
        DCP.a2d.to(device), DCP.d2a.to(device)
        DCP.a2e.to(device), DCP.e2a.to(device)
        DCP.c2d.to(device), DCP.d2c.to(device)
        DCP.c2e.to(device), DCP.e2c.to(device)
        DCP.d2e.to(device), DCP.e2d.to(device)

        if config['type'] == 'CG':
            acc, precision, f_measure = DCP.train_completegraph_supervised(config, logger, accumulated_metrics,
                                                                           x1_train, x2_train, x3_train, x4_train,
                                                                           x5_train, x1_test,
                                                                           x2_test, x3_test, x4_test, x5_test,
                                                                           labels_train,
                                                                           labels_test, optimizer, device)

            fold_acc.append(acc)
            fold_precision.append(precision)
            fold_f_measure.append(f_measure)
            print(time.time() - start)

        logger.info('--------------------Training over--------------------')
        acc, precision, f_measure = cal_classify(logger, fold_acc, fold_precision, fold_f_measure)


if __name__ == '__main__':
    main()
