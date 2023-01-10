# coding = utf-8
"""
作者   : Hilbert
时间   :2022/8/2 14:48
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

# parameters
window_size = 192
sample_rate = 2
epochs = 250
lr = 1e-3
batch_size = 256
model_name = 'CNN1D'
sensor_num = 4

output_dir = fr'/public/home/yuwen_hilbert/Program/CALCE_battery/model/result/fully_charge/' \
             fr'{model_name}/window_size_{window_size}_sample_rate_{sample_rate}_epochs_' \
             fr'{epochs}_lr_{lr}_batchsize_{batch_size}'
mkdir(output_dir)

data_name = ['PL21']
input_path = '/public/home/yuwen_hilbert/Program/CALCE_battery/battery_data/data_curve'
TrainingDataset = ModelDataset(battery_ids=data_name, input_path=input_path, state='training', window_size=window_size,
                               sample_rate=sample_rate, fully_curve=False, norm=False)
training_dataloader = DataLoader(TrainingDataset, batch_size=batch_size, shuffle=True)

TestDataset = ModelDataset(battery_ids=data_name, input_path=input_path, state='training', window_size=window_size,
                           sample_rate=sample_rate, fully_curve=False, norm=False)
test_dataloader = DataLoader(TrainingDataset, batch_size=batch_size, shuffle=True)


for sample, tar in training_dataloader:
    print(sample.shape)
    print(tar.shape)