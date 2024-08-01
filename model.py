from __future__ import print_function, absolute_import, division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle
from loss import instance_contrastive_Loss, category_contrastive_loss
from utils import classify
# from utils.next_batch import next_batch_gt, next_batch

class Autoencoder(nn.Module):
    """AutoEncoder module that projects features to latent space
    将特征投射到潜在空间的自动编码器模块."""

    def __init__(self,
                 encoder_dim,
                 activation='relu',
                 batchnorm=True):
        """Constructor.

        Args:
          encoder_dim: Should be a list of ints, hidden sizes of
            encoder network, the last element is the size of the latent representation.
          activation: Including "sigmoid", "tanh", "relu", "leakyrelu". We recommend to
            simply choose relu.
          batchnorm: if provided should be a bool type. It provided whether to use the
            batchnorm in autoencoders.
        """
        # 这是一个Autoencoder（自动编码器）的类定义，用于将特征投射到潜在空间。
        # 构造函数（init）的参数包括：
        #
        # encoder_dim：一个整数列表，表示编码器网络的隐藏层大小。列表中的最后一个元素是潜在表示的大小。
        # activation：一个字符串，指定编码器和解码器网络中要使用的激活函数。可选的值有"sigmoid"、“tanh”、“relu"和"leakyrelu”。
        # batchnorm：一个布尔值，指示是否在自动编码器中使用批归一化。
        #
        # 类定义了两个主要组件：编码器和解码器。编码器被定义为一系列线性层，后面跟着一个激活函数和可选的批归一化。解码器以类似的方式定义，但是层的大小是反向的。
        # 自动编码器的前向传播通过将输入数据通过编码器，然后通过解码器来定义。
        # 需要注意的是，对编码器的输出应用了softmax激活函数，这在自动编码器中是不寻常的。这可能是一个错误或特定于给定任务的设计选择。
        super(Autoencoder, self).__init__()
        self._dim = len(encoder_dim) - 1
        self._activation = activation
        self._batchnorm = batchnorm
        # 这部分代码调用了父类（nn.Module）的构造函数，并初始化了一些属性。
        # super(Autoencoder, self).__init__() 调用了父类的构造函数，即 nn.Module 的构造函数。这是必须的，因为我们定义的 Autoencoder 类是继承自 nn.Module 类的子类。这样可以确保我们的自动编码器类继承了 nn.Module 类的一些属性和方法。
        # self._dim = len(encoder_dim) - 1 初始化了一个名为 _dim 的实例量，它表示编码器网络的层数。由于 encoder_dim 是一个列表，其中的元素表示每个隐藏层的大小，所以 len(encoder_dim) - 1 的结果就是编码器网络的层数。
        # self._activation = activation 初始化了一个名为 _activation 的实例变量，它表示要在编码器和解码器网络中使用的激活函数。这个值是通过构造函数的参数 activation 来传递的。
        # self._batchnorm = batchnorm 初始化了一个名为 _batchnorm 的实例变量，它表示是否在自动编码器中使用批归一化。这个值是通过构造函数的参数 batchnorm 来传递的。
        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(
                nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
            if i < self._dim - 1:
                if self._batchnorm:
                    encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1]))
                if self._activation == 'sigmoid':
                    encoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    encoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    encoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        encoder_layers.append(nn.Softmax(dim=1))
        self._encoder = nn.Sequential(*encoder_layers)
        # 这部分代码定义了编码器的网络层，并将其存储在 self._encoder 变量中。
        # encoder_layers 是一个空列表，用于存储编码器的网络层。
        # 通过一个循环，将线性层 nn.Linear(encoder_dim[i], encoder_dim[i + 1]) 添加到 encoder_layers 中。这些线性层的输入和输出维度由 encoder_dim 列表中的元素确定，每个元素表示相应隐藏层的大小。
        # 在每个线性层之后，根据激活函数和批归一化的设置，可能会添加额外的层。如果 i 小于 self._dim - 1，则在当前线性层之后添加激活函数和批归一化层。
        #
        # 如果 self._batchnorm 为真，则在当前线性层之后添加批归一化层 nn.BatchNorm1d(encoder_dim[i + 1])。
        # 根据 self._activation 的值，选择相应的激活函数，并将其添加到 encoder_layers 中。
        #
        # 循环结束后，将一个具有 softmax 激活函数的层 nn.Softmax(dim=1) 添加到 encoder_layers 的末尾。这是不寻常的，因为在自动编码器中一般不使用 softmax 激活函数。
        # 最后，通过 nn.Sequential(*encoder_layers) 创建一个顺序的模型，并将其赋值给 self._encoder 变量。这个顺序的模型将按照列表中的顺序依次应用各个层。
        decoder_dim = [i for i in reversed(encoder_dim)]
        decoder_layers = []
        for i in range(self._dim):
            decoder_layers.append(
                nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            if self._batchnorm:
                decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
            if self._activation == 'sigmoid':
                decoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                decoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                decoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._decoder = nn.Sequential(*decoder_layers)
    # 这部分代码定义了解码器的网络层，并将其存储在 self._decoder 变量中。
    # 首先，通过将 encoder_dim 列表中的元素逆序生成了 decoder_dim 列表。这样做是为了确保解码器的网络层大小与编码器的网络层大小相对应。
    # 然后，通过一个循环，将线性层 nn.Linear(decoder_dim[i], decoder_dim[i + 1]) 添加到 decoder_layers 中。这些线性层的输入和输出维度由 decoder_dim 列表中的元素确定。
    # 在每个线性层之后，根据激活函数和批归一化的设置，可能会添加额外的层。
    #
    # 如果 self._batchnorm 为真，则在当前线性层之后添加批归一化层 nn.BatchNorm1d(decoder_dim[i + 1])。
    # 根据 self._activation 的值，选择相应的激活函数，并将其添加到 decoder_layers 中。
    #
    # 最后，通过 nn.Sequential(*decoder_layers) 创建一个顺序的模型，并将其赋值给 self._decoder 变量。这个顺序的模型将按照列表中的顺序依次应用各个层。
    def encoder(self, x):
        """Encode sample features.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [n_nodes, latent_dim] float tensor, representation Z.
        """
        latent = self._encoder(x)
        return latent
    # 这部分代码定义了一个名为 encoder 的方法，用于对输入样本特征进行编码。
    # 该方法接收一个输入张量 x，其形状为 [num, feat_dim]，其中 num 是样本数量，feat_dim 是特征维度。
    # 在方法内部，通过调用 _encoder 属性（即编码器网络）来对输入进行编码。编码器网络会将输入张量映射到一个潜在表示（latent representation）。
    # 最后，将编码后的潜在表示 latent 返回。其形状为 [n_nodes, latent_dim]，其中 n_nodes 是样本数量，latent_dim 是潜在表示的维度。
    def decoder(self, latent):
        """Decode sample features.

            Args:
              latent: [num, latent_dim] float tensor, representation Z.

            Returns:
              x_hat: [n_nodes, feat_dim] float tensor, reconstruction x.
        """
        x_hat = self._decoder(latent)
        return x_hat
    # 这部分代码定义了一个名为 decoder 的方法，用于对潜在表示进行解码，生成重构的样本特征。
    # 该方法接收一个输入张量 latent，其形状为 [num, latent_dim]，其中 num 是样本数量，latent_dim 是潜在表示的维度。
    # 在方法内部，通过调用 _decoder 属性（即解码器网络）来对潜在表示进行解码。解码器网络会将潜在表示映射回原始的样本特征空间，生成重构的样本特征。
    # 最后，将重构的样本特征 x_hat 返回。其形状为 [n_nodes, feat_dim]，其中 n_nodes 是样本数量，feat_dim 是特征维度。
    def forward(self, x):
        """Pass through autoencoder.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [num, latent_dim] float tensor, representation Z.
              x_hat:  [num, feat_dim] float tensor, reconstruction x.
        """
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        return x_hat, latent
# 这部分代码定义了一个名为 forward 的方法，用于通过自动编码器进行前向传播。
# 该方法接收一个输入张量 x，其形状为 [num, feat_dim]，其中 num 是样本数量，feat_dim 是特征维度。
# 在方法内部，首先调用 encoder 方法对输入进行编码，得到潜在表示 latent。
# 然后，调用 decoder 方法对潜在表示进行解码，生成重构的样本特征 x_hat。
# 最后，将重构的样本特征 x_hat 和潜在表示 latent 返回。它们的形状分别为 [num, feat_dim] 和 [num, latent_dim]，
# 其中 num 是样本数量，feat_dim 是特征维度，latent_dim 是潜在表示的维度。

class Prediction(nn.Module):
    """Dual prediction module that projects features from corresponding latent space."""

    def __init__(self,
                 prediction_dim,
                 activation='relu',
                 batchnorm=True):
        """Constructor.

        Args:
          prediction_dim: Should be a list of ints, hidden sizes of
            prediction network, the last element is the size of the latent representation of autoencoder.
          activation: Including "sigmoid", "tanh", "relu", "leakyrelu". We recommend to
            simply choose relu.
          batchnorm: if provided should be a bool type. It provided whether to use the
            batchnorm in autoencoders.
        """
        super(Prediction, self).__init__()

        self._depth = len(prediction_dim) - 1
        self._activation = activation
        self._prediction_dim = prediction_dim
        # 这是一个名为 Prediction 的类定义，用于进行特征的预测（或投影）。
        # 构造函数（__init__）的参数包括：
        # prediction_dim：一个整数列表，表示预测网络的隐藏层大小。列表中的最后一个元素是自动编码器的潜在表示的大小。
        # activation：一个字符串，指定预测网络中要使用的激活函数。可选的值有"sigmoid"、“tanh”、“relu"和"leakyrelu”。
        # batchnorm：一个布尔值，指示是否在预测网络中使用批归一化。
        # 在构造函数中，调用了父类（nn.Module）的构造函数，并初始化了一些属性。
        # self._depth = len(prediction_dim) - 1 初始化了一个名为 _depth 的实例变量，它表示预测网络的层数。由于 prediction_dim 是一个列表，
        # 其中的元素表示每个隐藏层的大小，所以 len(prediction_dim) - 1 的结果就是预测网络的层数。
        # self._activation = activation 初始化了一个名为 _activation 的实例变量，它表示要在预测网络中使用的激活函数。这个值是通过构造函数的参数 activation 来传递的。
        # self._prediction_dim = prediction_dim 初始化了一个名为 _prediction_dim 的实例变量，它表示预测网络的隐藏层大小。
        # 这个值是通过构造函数的参数 prediction_dim 来传递的。
        encoder_layers = []
        for i in range(self._depth):
            encoder_layers.append(
                nn.Linear(self._prediction_dim[i], self._prediction_dim[i + 1]))
            if batchnorm:
                encoder_layers.append(nn.BatchNorm1d(self._prediction_dim[i + 1]))
            if self._activation == 'sigmoid':
                encoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                encoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                encoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._encoder = nn.Sequential(*encoder_layers)
        # 这段代码是一个类的初始化方法，它创建了一个神经网络编码器（encoder）。
        # 首先，它创建了一个空列表encoder_layers，用于存储编码器的所有层。
        # 然后，通过一个循环，根据self._depth的值，逐个创建线性层（nn.Linear），并将它们添加到encoder_layers中。
        # 每个线性层的输入维度为self._prediction_dim[i]，输出维度为self._prediction_dim[i + 1]。这个循环创建了编码器的所有线性层。
        # 接下来，如果batchnorm为真，则在编码器的最后一层添加一个批归一化层（nn.BatchNorm1d）。
        # 然后，根据self._activation的值，通过条件语句选择不同的激活函数，并将它们添加到encoder_layers中。可选的激活函数有sigmoid、leakyrelu、tanh和relu。
        # 最后，通过nn.Sequential函数将encoder_layers列表中的层按顺序组合成一个序列，并将其赋值给self._encoder。这个序列就是整个编码器网络。
        decoder_layers = []
        for i in range(self._depth, 0, -1):
            decoder_layers.append(
                nn.Linear(self._prediction_dim[i], self._prediction_dim[i - 1]))
            if i > 1:
                if batchnorm:
                    decoder_layers.append(nn.BatchNorm1d(self._prediction_dim[i - 1]))
                if self._activation == 'sigmoid':
                    decoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    decoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    decoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        decoder_layers.append(nn.Softmax(dim=1))
        self._decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        """Data recovery by prediction.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [num, latent_dim] float tensor.
              output:  [num, feat_dim] float tensor, recovered data.
        """
        latent = self._encoder(x)
        output = self._decoder(latent)
        return output, latent
