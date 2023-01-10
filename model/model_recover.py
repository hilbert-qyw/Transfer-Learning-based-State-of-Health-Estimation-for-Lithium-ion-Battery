# coding = utf-8
"""
作者   : Hilbert
时间   :2022/8/19 9:44
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
from model_lib import *
from informer import *

# parameters
epochs = 200
model_name = 'CNN1D'
sensor_num = 4

batch_size=128
lr=5.949533369355227e-05
dropout=0.2
window_size=160
sample_rate=2
d_model=128
n_head=1
encoder_layer=2
cnn_kernel_size=2
reg_size=16


# last_output_dir = fr'/public/home/yuwen_hilbert/Program/CALCE_battery/model/result/fully_charge/' \
#                  fr'{model_name}/window_size_{window_size}_sample_rate_{sample_rate}_epochs_' \
#                  fr'{250}_lr_{lr}_batchsize_{batch_size}'
last_output_dir = fr'/public/home/yuwen_hilbert/Program/CALCE_battery/model/result/shallow_charge/' \
                  'CNN1D/time_0.00038255850268058_window_size_160_sample_rate_2_' \
                  'stride_1_epochs_150_lr_0.0001_batchsize_16/'


if model_name == 'CNN1D':
    net = CNN1D(window_size=window_size, sensor_num=sensor_num, kernel_size=3, linear_hidden_size_1=128,
                linear_hidden_size_2=64)
elif model_name == 'LSTM':
    net = LSTM(input_timestep=window_size, sensor_num=sensor_num)
    print(1)
elif model_name == 'CNN_attention':
    net = CNN_attention(window_size=window_size, sensor_num=sensor_num, kernel_size=2, linear_hidden_size_1=128,
                        linear_hidden_size_2=32)
elif model_name == 'Informer_CNN':
    net = Informer_CNN(enc_in=sensor_num, c_out=1, window_size=window_size, factor=5,
                       d_model=d_model, n_heads=n_head,
                       e_layers=encoder_layer, d_layers=encoder_layer-1, d_ff=512,
                       dropout=dropout,
                       attn='prob', embed='fixed', freq='h', activation='gelu', output_attention=False,
                       cnn_kernel_size=cnn_kernel_size, reg_size=reg_size,
                       distil=True, mix=True, device=torch.device('cuda:0'))

# 导入部分参数
# net_dict = net.state_dict()
# pretrained_dict = torch.load(os.path.join(last_output_dir, f'best_model_44.pt'))
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'cnn1d' in k}
# net_dict.update(pretrained_dict)
# net.load_state_dict(net_dict)
# net.load_state_dict(torch.load(os.path.join(last_output_dir, f'last_model.pt')))


criterion = nn.MSELoss()
# test
data_name = ['PL21']
input_path = '/public/home/yuwen_hilbert/Program/CALCE_battery/battery_data/data_curve'
TestDataset = CALCEModelDataset(battery_ids=data_name, input_path=input_path, state='test', window_size=window_size,
                                sample_rate=sample_rate, period_rate=1, stride=1, fully_curve=True, norm=False)
input_features, targets = TestDataset.moving_window()

tester = Tester(model=net, test_data=[input_features, targets], criterion=criterion,input_path=last_output_dir,
                saving_path=last_output_dir+fr'/PL21(shallow_model)',
                mode='last', best_epoch=59)
tester.test()

data_name = ['PL23']
input_path = '/public/home/yuwen_hilbert/Program/CALCE_battery/battery_data/data_curve'
TestDataset = CALCEModelDataset(battery_ids=data_name, input_path=input_path, state='test', window_size=window_size,
                                sample_rate=sample_rate, period_rate=1, stride=1, fully_curve=True, norm=False)
input_features, targets = TestDataset.moving_window()

tester = Tester(model=net, test_data=[input_features, targets], criterion=criterion,input_path=last_output_dir,
                saving_path=last_output_dir+fr'/PL23(shallow_model)',
                mode='last', best_epoch=59)
tester.test()