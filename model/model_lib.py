# coding = utf-8
"""
作者   : Hilbert
时间   :2022/7/24 16:23
"""
import sys
import os
from warnings import simplefilter

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]  # 上一级目录
PathProject = os.path.split(rootPath)[0]
sys.path.append(rootPath)
sys.path.append(PathProject)
simplefilter(action='ignore', category=Warning)
simplefilter(action='ignore', category=FutureWarning)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class CNN1D(nn.Module):
    def __init__(self,
                 window_size,
                 sensor_num,
                 kernel_size=2,
                 linear_hidden_size_1=256,
                 linear_hidden_size_2=32
                 ):
        super(CNN1D, self).__init__()
        filter_1 = sensor_num * 2
        filter_2 = sensor_num * 4
        filter_3 = sensor_num * 8
        filter_4 = sensor_num * 16

        self.cnn1d = nn.Sequential(nn.Conv1d(in_channels=sensor_num, out_channels=filter_1, kernel_size=kernel_size,
                                             padding='same'),
                                   nn.ReLU(),
                                   nn.MaxPool1d(kernel_size=2, stride=2),
                                   nn.Conv1d(in_channels=filter_1, out_channels=filter_2, kernel_size=kernel_size,
                                             padding='same'),
                                   nn.ReLU(),
                                   nn.MaxPool1d(kernel_size=2, stride=2),
                                   nn.Conv1d(in_channels=filter_2, out_channels=filter_3, kernel_size=kernel_size,
                                             padding='same'),
                                   nn.ReLU(),
                                   nn.MaxPool1d(kernel_size=2, stride=2),
                                   # nn.ReLU(),
                                   nn.Conv1d(in_channels=filter_3, out_channels=filter_4, kernel_size=kernel_size,
                                             padding='same'),
                                   nn.MaxPool1d(kernel_size=2, stride=2),
                                   )

        self.regression = nn.Sequential(nn.Linear(in_features=sensor_num * window_size, out_features=linear_hidden_size_1),
                                        # nn.ReLU(),
                                        # nn.Dropout(),
                                        nn.Linear(in_features=linear_hidden_size_1, out_features=linear_hidden_size_2),
                                        nn.ReLU(),
                                        # nn.Dropout(),
                                        nn.Linear(in_features=linear_hidden_size_2, out_features=1))

    def forward(self, x):
        # batchsize, sequence, channels
        x = x.permute(0, 2, 1)
        # print(x.shape)  # batch_size, channels, sequence

        conv_out = self.cnn1d(x)
        reg_input = conv_out.view(conv_out.shape[0], -1)
        out = self.regression(reg_input)

        return out.view(-1)


class SourceTransferCNN1D(nn.Module):
    def __init__(self,
                 window_size,
                 sensor_num,
                 kernel_size=2,
                 linear_hidden_size_1=256,
                 linear_hidden_size_2=32,
                 transfer_fcn_size=32,
                 lstm_layers=2,
                 ):
        super(SourceTransferCNN1D, self).__init__()
        filter_1 = sensor_num * 2
        filter_2 = sensor_num * 4
        filter_3 = sensor_num * 8
        filter_4 = sensor_num * 16

        # self.source_transfer = nn.Sequential(nn.Linear(in_features=window_size, out_features=window_size),
        #                                      nn.ReLU(),
        #                                      nn.Linear(in_features=window_size, out_features=window_size),
        #                                      )

        self.source_transfer = nn.LSTM(input_size=sensor_num, hidden_size=transfer_fcn_size,
                                       num_layers=lstm_layers,
                                       batch_first=True)
        self.fcn = nn.Linear(in_features=transfer_fcn_size, out_features=sensor_num)

        self.cnn1d = nn.Sequential(nn.Conv1d(in_channels=sensor_num, out_channels=filter_1, kernel_size=kernel_size,
                                             padding='same'),
                                   nn.ReLU(),
                                   nn.MaxPool1d(kernel_size=2, stride=2),
                                   nn.Conv1d(in_channels=filter_1, out_channels=filter_2, kernel_size=kernel_size,
                                             padding='same'),
                                   nn.ReLU(),
                                   nn.MaxPool1d(kernel_size=2, stride=2),
                                   nn.Conv1d(in_channels=filter_2, out_channels=filter_3, kernel_size=kernel_size,
                                             padding='same'),
                                   nn.ReLU(),
                                   nn.MaxPool1d(kernel_size=2, stride=2),
                                   # nn.ReLU(),
                                   nn.Conv1d(in_channels=filter_3, out_channels=filter_4, kernel_size=kernel_size,
                                             padding='same'),
                                   nn.MaxPool1d(kernel_size=2, stride=2),
                                   )

        self.regression = nn.Sequential(nn.Linear(in_features=sensor_num * window_size, out_features=linear_hidden_size_1),
                                        # nn.ReLU(),
                                        # nn.Dropout(),
                                        nn.Linear(in_features=linear_hidden_size_1, out_features=linear_hidden_size_2),
                                        nn.ReLU(),
                                        # nn.Dropout(),
                                        nn.Linear(in_features=linear_hidden_size_2, out_features=1))

    def forward(self, x):
        new_x, _ = self.source_transfer(x)
        x = self.fcn(new_x)

        # new_x = self.source_transfer(x)
        # x = x + new_x
        # b = x[0].detach().numpy()
        # c = new_x[0].detach().numpy()
        # d = x[0].detach().numpy()
        # batchsize, sequence, channels
        x = x.permute(0, 2, 1)
        # print(x.shape)  # batch_size, channels, sequence
        # x = self.source_transfer(x)
        conv_out = self.cnn1d(x)
        reg_input = conv_out.view(conv_out.shape[0], -1)
        out = self.regression(reg_input)

        return out.view(-1)


class LSTM(nn.Module):
    def __init__(self,
                 input_timestep,
                 sensor_num,
                 lstm_hidden_layers=3,
                 lstm_hidden_size=32,
                 linear_hidden_size_1=64,
                 linear_hidden_size_2=32):

        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=sensor_num, hidden_size=lstm_hidden_size, num_layers=lstm_hidden_layers,
                            batch_first=True)

        self.regression = nn.Sequential(nn.Linear(in_features=lstm_hidden_size, out_features=linear_hidden_size_1),
                                        nn.ReLU(),
                                        nn.Linear(in_features=linear_hidden_size_1, out_features=linear_hidden_size_2),
                                        nn.ReLU(),
                                        nn.Linear(in_features=linear_hidden_size_2, out_features=1))
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        reg_input = lstm_out[:, -1, :]
        out = self.regression(reg_input)

        return out.view(-1)


class CNN_LSTM(nn.Module):
    def __init__(self,
                 input_timestep,
                 sensor_num=34,
                 predict_sensor=9,
                 lstm_hidden_layers=4,
                 lstm_hidden_size=128,
                 linear_hidden_size_1=256,
                 linear_hidden_size_2=64):
        super(CNN_LSTM, self).__init__()
        self.input_timestep = input_timestep
        filter_1 = sensor_num * 2
        filter_2 = sensor_num * 4
        filter_3 = sensor_num * 8

        self.cnn1d = nn.Sequential(nn.Conv1d(in_channels=sensor_num, out_channels=filter_1, kernel_size=2, padding='same'),
                                   nn.ReLU(),
                                   nn.MaxPool1d(kernel_size=2, stride=2),
                                   nn.Conv1d(in_channels=filter_1, out_channels=filter_2, kernel_size=2, padding='same'),
                                   nn.ReLU(),
                                   nn.MaxPool1d(kernel_size=2, stride=2),
                                   nn.Conv1d(in_channels=filter_2, out_channels=filter_3, kernel_size=2, padding='same'),
                                   nn.ReLU(),
                                   nn.MaxPool1d(kernel_size=2, stride=2)
                                   )

        self.lstm = nn.LSTM(input_size=sensor_num, hidden_size=lstm_hidden_size, num_layers=lstm_hidden_layers,
                            batch_first=True)

        self.regression = nn.Sequential(nn.Linear(in_features=lstm_hidden_size, out_features=linear_hidden_size_1),
                                        nn.ReLU(),
                                        nn.Linear(in_features=linear_hidden_size_1, out_features=linear_hidden_size_2),
                                        nn.ReLU(),
                                        nn.Linear(in_features=linear_hidden_size_2, out_features=predict_sensor))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        conv_x = self.cnn1d(x)
        lstm_input = conv_x.view(conv_x.shape[0], self.input_timestep, -1)
        lstm_out, _ = self.lstm(lstm_input)
        reg_input = lstm_out[:, -1, :]
        out = self.regression(reg_input)

        return out


class CNN_attention(nn.Module):
    def __init__(self,
                 window_size,
                 sensor_num,
                 kernel_size=2,
                 attention_hidden_size=64,
                 linear_hidden_size_1=128,
                 linear_hidden_size_2=32,
                 ):
        super(CNN_attention, self).__init__()
        self.input_timestep = window_size
        self.sensor_num = sensor_num
        filter_1 = sensor_num * 2
        filter_2 = sensor_num * 4
        filter_3 = sensor_num * 8

        self.cnn1d = nn.Sequential(nn.Conv1d(in_channels=sensor_num, out_channels=filter_1,
                                             kernel_size=kernel_size, padding='same'),
                                   nn.BatchNorm1d(filter_1),
                                   nn.ReLU(),
                                   nn.MaxPool1d(kernel_size=2, stride=2),
                                   nn.Conv1d(in_channels=filter_1, out_channels=filter_2,
                                             kernel_size=kernel_size, padding='same'),
                                   nn.BatchNorm1d(filter_2),
                                   nn.ReLU(),
                                   nn.MaxPool1d(kernel_size=2, stride=2),
                                   nn.Conv1d(in_channels=filter_2, out_channels=filter_3,
                                             kernel_size=kernel_size, padding='same'),
                                   # nn.ReLU(),
                                   nn.BatchNorm1d(filter_3),
                                   nn.MaxPool1d(kernel_size=2, stride=2)
                                   )

        self.trans_linear = nn.Linear(in_features=int(window_size*sensor_num), out_features=window_size*sensor_num)
        self.attention_layer_1 = nn.Linear(in_features=sensor_num, out_features=attention_hidden_size)
        self.attention_layer_2 = nn.Linear(in_features=attention_hidden_size, out_features=1)
        self.regression = nn.Sequential(nn.Linear(in_features=sensor_num, out_features=linear_hidden_size_1),
                                        # nn.ReLU(),
                                        # nn.Dropout(),
                                        nn.Linear(in_features=linear_hidden_size_1, out_features=linear_hidden_size_2),
                                        nn.ReLU(),
                                        # nn.Dropout(),
                                        nn.Linear(in_features=linear_hidden_size_2, out_features=1),
                                        )
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        conv_x = self.cnn1d(x)
        conv_x = conv_x.view((conv_x.shape[0], -1))
        atten_input = self.relu(self.trans_linear(conv_x))
        # atten_input = self.trans_linear(conv_x)
        atten_input = atten_input.view((atten_input.shape[0], self.input_timestep, self.sensor_num))
        weights = self.attention_weights(atten_input)
        atten_output = self.sumpooling(atten_input, weights)
        # atten_output shape: [bs, sensor_num, input_timestep]
        out = self.regression(atten_output)
        return out.view(-1)

    def attention_weights(self, atten_input):
        score_hidden = self.tanh(self.attention_layer_1(atten_input))
        score_output = self.attention_layer_2(score_hidden)
        weights = self.softmax(score_output.view(score_output.shape[0], score_output.shape[1]))
        return weights

    def sumpooling(self, atten_input, weights):
        atten_input = atten_input.permute(0, 2, 1)
        weights = weights.view((weights.shape[0], -1, 1))
        atten_output = torch.matmul(atten_input, weights)

        return atten_output.squeeze()


class MCNN(nn.Module):
    def __init__(self, ts_shape: tuple, n_classes: int, pool_factor: int,
                 kernel_size: float or int, transformations: dict):
        """
        Multi-Scale Convolutional Neural Network for Time Series Classification - Cui et al. (2016).

        Args:
          ts_shape (tuple):           shape of the time series, e.g. (1, 9) for uni-variate time series
                                      with length 9, or (3, 9) for multivariate time series with length 9
                                      and three features
          n_classes (int):            number of classes
          pool_factor (int):          length of feature map after max pooling, usually in {2,3,5}
          kernel_size (int or float): filter size for convolutional layers, usually ratio in {0.05, 0.1, 0.2}
                                      times the length of time series
          transformations (dict):     dictionary with key value pairs specifying the transformations
                                      in the format 'name': {'class': <TransformClass>, 'params': <parameters>}
        """
        assert len(ts_shape) == 2, "Expecting shape in format (n_channels, seq_len)!"

        super(MCNN, self).__init__()

        self.ts_shape = ts_shape
        self.n_classes = n_classes
        self.pool_factor = pool_factor
        self.kernel_size = int(self.ts_shape[1] * kernel_size) if kernel_size < 1 else int(kernel_size)

        self.loss = nn.CrossEntropyLoss

        # layer settings
        self.local_conv_filters = 256
        self.local_conv_activation = nn.ReLU  # nn.Sigmoid in original implementation
        self.full_conv_filters = 256
        self.full_conv_activation = nn.ReLU  # nn.Sigmoid in original implementation
        self.fc_units = 256
        self.fc_activation = nn.ReLU  # nn.Sigmoid in original implementation

        # setup branches
        self.branches = self._setup_branches(transformations)
        self.n_branches = len(self.branches)

        # full convolution
        in_channels = self.local_conv_filters * self.n_branches
        # kernel shouldn't exceed the length (length is always pool factor?)
        full_conv_kernel_size = int(min(self.kernel_size, int(self.pool_factor)))
        self.full_conv = nn.Conv1d(in_channels, self.full_conv_filters,
                                   kernel_size=full_conv_kernel_size,
                                   padding='same')
        pool_size = 1
        self.full_conv_pool = nn.MaxPool1d(pool_size)

        # fully-connected
        self.flatten = nn.Flatten()
        in_features = int(self.pool_factor * self.full_conv_filters)
        self.fc = nn.Linear(in_features, self.fc_units, bias=True)

        # softmax output
        self.output = nn.Linear(self.fc_units, self.n_classes, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        xs = [self.branches[idx](x) for idx in range(self.n_branches)]
        x = torch.cat(xs, dim=1)

        x = self.full_conv(x)
        x = self.full_conv_activation()(x)
        x = self.full_conv_pool(x)

        x = self.flatten(x)
        x = self.fc(x)
        x = self.fc_activation()(x)

        x = self.output(x)

        return x

    def _build_local_branch(self, name: str, transform: nn.Module, params: list):
        """
        Build transformation and local convolution branch.

        Args:
          name (str):   Name of the branch.
          transform (nn.Module):  Transformation class applied in this branch.
          params (list):   Parameters for the transformation, with the first parameter always being the input shape.
        Returns:
          branch:   Sequential model containing transform, local convolution, activation, and max pooling.
        """
        branch = nn.Sequential()
        # transformation
        # instance
        branch.add_module(name + '_transform', transform(*params))
        # local convolution
        branch.add_module(name + '_conv', nn.Conv1d(self.ts_shape[0], self.local_conv_filters,
                                                    kernel_size=self.kernel_size, padding='same'))
        branch.add_module(name + '_activation', self.local_conv_activation())
        # local max pooling (ensure that outputs all have length equal to pool factor)
        pool_size = int(int(branch[0].output_shape[1]) / self.pool_factor)
        assert pool_size > 1, "ATTENTION: pool_size can not be 0 or 1, as the lengths are then not equal" \
                              "for concatenation!"
        branch.add_module(name + '_pool', nn.MaxPool1d(pool_size))  # default stride equal to pool size

        return branch

    def _setup_branches(self, transformations: dict):
        """
        Setup all branches for the local convolution.

        Args:
          transformations:  Dictionary containing the transformation classes and parameter settings.
        Returns:
          branches: List of sequential models with local convolution per branch.
        """
        branches = []
        for transform_name in transformations:
            transform_class = transformations[transform_name]['class']
            parameter_list = transformations[transform_name]['params']

            # create transform layer for each parameter configuration
            if parameter_list:
                for param in parameter_list:
                    if np.isscalar(param):
                        # 判断是否为标量
                        name = transform_name + '_' + str(param)
                        branch = self._build_local_branch(name, transform_class, [self.ts_shape, param])
                    else:
                        branch = self._build_local_branch(transform_name, transform_class,
                                                          [self.ts_shape] + list(param))
                    branches.append(branch)
            else:
                branch = self._build_local_branch(transform_name, transform_class, [self.ts_shape])
                branches.append(branch)

        return torch.nn.ModuleList(branches)


# TCN
# reference: https://github.com/hilbert-qyw/TCN/blob/master/TCN/tcn.py
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        """
        其实这就是一个裁剪的模块，裁剪多出来的padding
        将后边多扩展的大小为padding的长度给删除，通过增加padding方式对卷积后的张量做切片而实现因果卷积（只能补在一边，不能补在另一边)
        tensor.contiguous() 会返回有连续内存的相同张量， 有些tensor并不是占用一整块内存，而是由不同的数据块组成
        而tensor的view()操作依赖于内存是整块，这时候就需要执行contiguous()这个函数，把tensor变成在内存中连续分布的形式
        """
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size


    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        相当于一个Residual block

        :param n_inputs: int, 输入通道数
        :param n_outputs: int, 输出通道数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长，一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """

        super(TemporalBlock, self).__init__()

        # 定义第一个扩散卷积层，扩散是dilation
        # 重参数将权重向量w用了两个独立的参数表示其幅度和方向，加速收敛
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))

        # 根据第一个卷积层的输出与padding大小实现因果卷积
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        # 在先前的输出结果上添加激活函数与dropout完成第一个卷积

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        # padding保证了输入序列与输出序列的长度相同，且padding只对序列的一边进行操作
        # 卷积前后的通道不一定一致
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 将卷积模块所有部件拼接起来
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        # point-wise 1*1 卷积，实现通道融合，将通道变换至输入相同，实现残差连接
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # 输入x进行空洞卷积、因果卷积
        out = self.net(x)

        # 输入x进行通道融合，使两个路径输出的数据维度相同，从而进行残差连接
        res = x if self.downsample is None else self.downsample(x)

        # 残差连接并返回
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self,
                 num_inputs,
                 num_channels,
                 kernel_size=2,
                 dropout=0,
                 linear_hidden_1=256,
                 linear_hidden_2=64,
                 predict_sensor=9):
        """
        :param num_inputs: int， 输入通道数
        :param num_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """

        super(TCN, self).__init__()
        layers = []
        # num_channels 为各层卷积运算的输出通道数或卷积核数量
        # num_channels 的长度即需要执行的卷积层数量
        num_levels = len(num_channels)
        # 扩张系数随网络层级的增加而成指数级增加，增大感受野并不丢弃任何输入序列的元素
        for i in range(num_levels):
            dilation_size = 2 ** i      # dilation_size 根据层级数指数增加
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            # 从num_channels中抽取每一个残差模块的输入通道数与输出通道数
            # 调用残差模块
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
        # 将所有残差模块堆叠起来组成一个深度卷积网络
        self.network = nn.Sequential(*layers)
        # shape: [bs, num_channels, seq_len]
        self.regression = nn.Sequential(nn.Linear(in_features=num_channels[-1], out_features=linear_hidden_1),
                                        nn.ReLU(),
                                        nn.Linear(in_features=linear_hidden_1, out_features=linear_hidden_2,),
                                        nn.ReLU(),
                                        nn.Linear(in_features=linear_hidden_2, out_features=predict_sensor),
                                        )

    def forward(self, x):
        """
        输入x的结构不同于RNN，一般RNN的size为(Batch, seq_len, channels)或者(seq_len, Batch, channels)，
        这里把seq_len放在channels后面，把所有时间步的数据拼起来，当做Conv1d的输入尺寸，实现卷积跨时间步的操作，
        很巧妙的设计。
        x: [bs, channels, seq_len]
        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        x = x.permute(0, 2, 1)
        tcn_out = self.network(x)
        out = self.regression(tcn_out[:, :, -1])

        return out


class TCN_attention(nn.Module):
    def __init__(self,
                 num_inputs,
                 num_channels,
                 kernel_size=2,
                 dropout=0,
                 attention_hidden_size=64,
                 linear_hidden_1=64,
                 linear_hidden_2=32,
                 predict_sensor=9,
                 ):

        super(TCN_attention, self).__init__()
        layers = []
        num_levels = len(num_channels)
        # 扩张系数随网络层级的增加而成指数级增加，增大感受野并不丢弃任何输入序列的元素
        for i in range(num_levels):
            dilation_size = 2 ** i  # dilation_size 根据层级数指数增加
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            # 从num_channels中抽取每一个残差模块的输入通道数与输出通道数
            # 调用残差模块
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
        # 将所有残差模块堆叠起来组成一个深度卷积网络
        self.network = nn.Sequential(*layers)
        # shape: [bs, num_channels, seq_len]

        self.attention_layer_1 = nn.Linear(in_features=num_channels[-1], out_features=attention_hidden_size)
        self.attention_layer_2 = nn.Linear(in_features=attention_hidden_size, out_features=1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        self.regression = nn.Sequential(nn.Linear(in_features=num_channels[-1], out_features=linear_hidden_1),
                                        nn.ReLU(),
                                        nn.Dropout(),
                                        nn.Linear(in_features=linear_hidden_1, out_features=linear_hidden_2),
                                        nn.ReLU(),
                                        nn.Dropout(),
                                        nn.Linear(in_features=linear_hidden_2, out_features=predict_sensor),
                                        )



    def forward(self, x):
        x = x.permute(0, 2, 1)
        # torch.Size([32, 34, 48])
        tcn_out = self.network(x)
        # shape: [bs, channels, seq_len]
        atten_input = tcn_out.permute(0, 2, 1)
        # shape: [bs, seq_len, channels]
        weights = self.attention_weights(atten_input)
        atten_output = self.sumpooling(atten_input, weights)
        out = self.regression(atten_output)

        return out

    def attention_weights(self, atten_input):
        score_hidden = self.tanh(self.attention_layer_1(atten_input))
        score_output = self.attention_layer_2(score_hidden)
        weights = self.softmax(score_output.squeeze())
        return weights

    def sumpooling(self, atten_input, weights):
        atten_input = atten_input.permute(0, 2, 1)
        weights = weights.view((weights.shape[0], -1, 1))
        atten_output = torch.matmul(atten_input, weights)

        return atten_output.squeeze()


class GRU_attention(nn.Module):
    def __init__(self,
                 input_timestep,
                 sensor_num,
                 predict_sensor,
                 gru_hidden_size=64,
                 gru_hidden_layers=3,
                 attention_hidden_size=32,
                 linear_hidden_size_1=64,
                 linear_hidden_size_2=32,
                 ):
        super(GRU_attention, self).__init__()

        self.gru = nn.GRU(input_size=sensor_num, hidden_size=gru_hidden_size, num_layers=gru_hidden_layers,
                            batch_first=True)

        self.regression = nn.Sequential(nn.Linear(in_features=gru_hidden_size, out_features=linear_hidden_size_1),
                                        nn.ReLU(),
                                        nn.Dropout(),
                                        nn.Linear(in_features=linear_hidden_size_1, out_features=linear_hidden_size_2),
                                        nn.ReLU(),
                                        nn.Dropout(),
                                        nn.Linear(in_features=linear_hidden_size_2, out_features=predict_sensor))
        self.relu = nn.ReLU()


        self.input_timestep = input_timestep
        self.sensor_num = sensor_num

        self.attention_layer_1 = nn.Linear(in_features=gru_hidden_size, out_features=attention_hidden_size)
        self.attention_layer_2 = nn.Linear(in_features=attention_hidden_size, out_features=1)
        self.regression = nn.Sequential(nn.Linear(in_features=gru_hidden_size, out_features=linear_hidden_size_1),
                                        nn.ReLU(),
                                        nn.Linear(in_features=linear_hidden_size_1, out_features=linear_hidden_size_2),
                                        nn.ReLU(),
                                        nn.Linear(in_features=linear_hidden_size_2, out_features=predict_sensor))
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # shape x: [bs, seq_len, sensors]
        gru_x, _ = self.gru(x)
        # shape: [bs, seq_len, gru_hidden_size]
        atten_input = gru_x
        weights = self.attention_weights(atten_input)
        atten_output = self.sumpooling(atten_input, weights)
        # atten_output shape: [bs, sensor_num, input_timestep]
        out = self.regression(atten_output)
        return out


    def attention_weights(self, atten_input):
        score_hidden = self.tanh(self.attention_layer_1(atten_input))
        score_output = self.attention_layer_2(score_hidden)
        weights = self.softmax(score_output.squeeze())
        return weights

    def sumpooling(self, atten_input, weights):
        atten_input = atten_input.permute(0, 2, 1)
        weights = weights.view((weights.shape[0], -1, 1))
        atten_output = torch.matmul(atten_input, weights)

        return atten_output.squeeze()


"""vision transformer"""
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # x --> embedding --> chunk

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        # [batch_size, num_heads, embed_dim_per_head, num_patches + 1]

        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        # 每个header的q和k相乘，除以√dim_k（相当于norm处理）
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale


        attn = self.attend(dots)    # 通过softmax处理（相当于对每一行的数据softmax）
        attn = self.dropout(attn)   # dropOut层

        out = torch.matmul(attn, v)
        # 得到的结果和V矩阵相乘（加权求和）
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]

        out = rearrange(out, 'b h n d -> b n (h d)')
        # 把head拼接,  [batch_size, num_patches + 1, total_embed_dim]

        # to_out 通过全连接进行映射
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3,
                 dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        # the width and height of patch
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        # must be integer
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # equals to the seq_len of LSTM
        num_patches = (image_height // patch_height) * (image_width // patch_width)

        # the dimension of embedding
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # 张量变换，将image-->patch
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # 注意，在vit模型中输入大小必须是固定的，高宽和设定值相同
        # img: [1, 3, 256, 256]
        x = self.to_patch_embedding(img)
        # x:[1, 64, 1024]  image --> patch (linear)


        b, n, _ = x.shape

        # self.cls_token, randomly initialize, repeat b times
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1) # [bs, num_patches + 1, dim]
        x += self.pos_embedding[:, :(n + 1)]
        # self.pos_embedding: [1, num_patches + 1, dim]
        # x: [bs, num_patches + 1, dim]

        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class TDRL_v4(nn.Module):
    def __init__(self, sensor_num, cycle_step, many2one=True):
        super(TDRL_v4, self).__init__()
        dim_atten = 32
        dim_output = 32
        filter_1 = 32
        filter_2 = 64
        filter_3 = 128
        filter_4 = 256
        self.sensor_num = sensor_num
        self.cycle_step = cycle_step
        self.conv_layer1 = nn.Conv1d(in_channels=sensor_num, out_channels=filter_1, kernel_size=2, padding=1)
        self.max_pool_1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv_layer2 = nn.Conv1d(in_channels=filter_1, out_channels=filter_2, kernel_size=2, padding=1)
        self.max_pool_2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv_layer3 = nn.Conv1d(in_channels=filter_2, out_channels=filter_3, kernel_size=2, padding=1)
        self.max_pool_3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv_layer4 = nn.Conv1d(in_channels=filter_3, out_channels=filter_4, kernel_size=2, padding=1)
        self.max_pool_4 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.conv_fc = nn.Linear(1024, self.sensor_num * self.cycle_step)
        self.dropout = nn.Dropout(p=0.5)
        self.layer_attention = nn.Linear(cycle_step*4, dim_atten)
        # self.layer_attention = nn.Linear(time_step*4, dim_atten)
        self.layer_atten_out = nn.Linear(dim_atten, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.layer_outhidden = nn.Linear(cycle_step, dim_output)
        self.layer_output = nn.Linear(dim_output, 1)

    def forward(self, input):
        """
        :param input: [None, cycle_step, sensor_num]
        :return:
        """
        # 1D CNN
        input = input.permute(0, 2, 1)
        # print(input.shape)
        conv1 = self.relu(self.conv_layer1(input))
        # print(conv1.shape)
        max_pool_1 = self.max_pool_1(conv1)
        # print(max_pool_1.shape)
        # conv2 = self.conv_layer2(max_pool_1)
        conv2 = self.relu(self.conv_layer2(max_pool_1))
        max_pool_2 = self.max_pool_2(conv2)
        # print(max_pool_2.shape)
        conv3 = self.conv_layer3(max_pool_2)
        # conv3 = self.relu(self.conv_layer3(max_pool_2))
        max_pool_3 = self.max_pool_3(conv3)
        # # print(max_pool_3.shape)
        # conv4 = self.conv_layer4(max_pool_3)
        # max_pool_4 = self.max_pool_4(conv4)
        conv_out = max_pool_3.view(max_pool_3.shape[0], -1)
        # print(conv_out.shape)
        # attention network
        atten_input = self.conv_fc(conv_out)
        # print(atten_input.shape)
        atten_input = atten_input.view(-1, self.sensor_num, self.cycle_step)
        atten_base = atten_input[:, 0, :]
        atten_base = atten_base.view(atten_base.shape[0], -1, atten_base.shape[1])
        atten_bases = atten_base.repeat(1, atten_input.shape[1], 1)
        score_att = self.attention_unit(atten_input, atten_bases)
        embed_pool = self.sumpooling(atten_input, score_att)
        # print(embed_pool.shape)
        out_hidden = self.relu(self.layer_outhidden(embed_pool))
        # print(out_hidden.shape)
        out = self.layer_output(out_hidden)
        # out = self.layer_output(out_hidden)
        out = out.flatten()
        return out, score_att, conv_out, atten_input, embed_pool

    def sumpooling(self, embed_his, score_att):
        '''
        :param embed_his: [None, cycle_step, dim_embed]
        :param score_att: [None, cycle_step]
        '''
        score_att = score_att.view((-1, score_att.shape[1], 1))
        embed_his = embed_his.permute((0, 2, 1))
        embed = torch.matmul(embed_his, score_att)
        return embed.view((-1, embed.shape[1]))

    def attention_unit(self, embed_his, embed_bases):
        '''
        :param embed_his: [None, cycle_step, dim_embed]
        :param embed_base: [None, 1, dim_embed]
        '''
        embed_concat = torch.cat([embed_his, embed_bases, embed_his - embed_bases, embed_his * embed_bases],
                                 dim=2)  # [None, cycle_step, dim_embed*4]
        hidden_att = self.tanh(self.layer_attention(embed_concat))
        # print(hidden_att.shape)
        score_att = self.layer_atten_out(hidden_att)
        # print(score_att.shape)
        score_att = self.softmax(score_att.view((-1, score_att.shape[1])))
        return score_att