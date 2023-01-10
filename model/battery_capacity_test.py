# coding = utf-8
"""
作者   : Hilbert
时间   :2022/7/25 15:02
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

from utils import *
from battery_data.CALCE_data_process.lib import *
from model_lib import *
import torch.nn as nn

# parameters
window_size = 192
sample_rate = 2
epochs = 200
lr = 1e-3
batch_size = 256
model_name = 'CNN1D'
sensor_num = 4

output_dir = fr'./result/fully_charge/{model_name}/window_size_{window_size}_sample_rate_{sample_rate}_epochs_' \
             fr'{epochs}_lr_{lr}_batchsize_{batch_size}'
mkdir(output_dir)


data_name = ['PL13']
input_path = '../battery_data/data_curve'
TestDataset = ModelDataset(battery_ids=data_name, input_path=input_path, state='test', window_size=window_size, sample_rate=sample_rate,
                           fully_curve=True, norm=False)
input_features, targets = TestDataset.inputs, TestDataset.outputs


if model_name == 'CNN1D':
    net = CNN1D(window_size=window_size, sensor_num=sensor_num, kernel_size=3, linear_hidden_size_1=64, linear_hidden_size_2=16)
elif model_name == 'LSTM':
    net = LSTM(input_timestep=window_size, sensor_num=sensor_num)
    print(1)

criterion = nn.MSELoss()

print("Start to test...")
tester = Tester(model=net, test_data=[input_features, targets], criterion=criterion, saving_path=output_dir,
                mode='best', best_epoch=32)
tester.test(mode='best')