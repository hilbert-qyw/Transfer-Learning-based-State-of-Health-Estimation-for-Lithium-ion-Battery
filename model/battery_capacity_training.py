# coding = utf-8
"""
作者   : Hilbert
时间   :2022/7/23 15:13
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
from informer import *
from scipy.stats import loguniform


# parameters
window_size = 160
sample_rate = 2
epochs = 150
lr = 1e-4
batch_size = 256
model_name = 'CNN1D'
sensor_num = 4
stride = 2
time = loguniform.rvs(1e-5, 1e-3, size=1)[0]

output_dir = fr'/public/home/yuwen_hilbert/Program/CALCE_battery/model/result/shallow_charge/' \
             fr'{model_name}/time_{time}_window_size_{window_size}_sample_rate_{sample_rate}_stride_{stride}_epochs_' \
             fr'{epochs}_lr_{lr}_batchsize_{batch_size}'
mkdir(output_dir)

data_name = ['PL11']
input_path = '/public/home/yuwen_hilbert/Program/CALCE_battery/battery_data/data_curve'
TrainingDataset = CALCEModelDataset(battery_ids=data_name, input_path=input_path, state='training', window_size=window_size,
                                    sample_rate=sample_rate, period_rate=1, stride=stride, fully_curve=True, norm=False)
training_dataloader = DataLoader(TrainingDataset, batch_size=batch_size, shuffle=True)

data_name = ['PL23']
input_path = '/public/home/yuwen_hilbert/Program/CALCE_battery/battery_data/data_curve'
ValDataset = CALCEModelDataset(battery_ids=data_name, input_path=input_path, state='val', window_size=window_size,
                               sample_rate=sample_rate, period_rate=1, stride=1, fully_curve=False, norm=False)
val_dataloader = DataLoader(ValDataset, batch_size=batch_size, shuffle=False)

if model_name == 'CNN1D':
    net = CNN1D(window_size=window_size, sensor_num=sensor_num, kernel_size=3, linear_hidden_size_1=128,
                linear_hidden_size_2=64)
elif model_name == 'LSTM':
    net = LSTM(input_timestep=window_size, sensor_num=sensor_num)
    print(1)
elif model_name == 'CNN_attention':
    net = CNN_attention(window_size=window_size, sensor_num=sensor_num, kernel_size=2, linear_hidden_size_1=128,
                        linear_hidden_size_2=32)
elif model_name == 'Informer':
    net = Informer(enc_in=4, c_out=1, factor=5, d_model=512, n_heads=8,
                   e_layers=3, d_layers=2, d_ff=512, dropout=0.0, attn='prob',
                   embed='fixed', freq='h', activation='gelu', output_attention=False,
                   lstm_hidden_size=256, lstm_hidden_layers=3, reg_size=32,
                   distil=True, mix=True, device=torch.device('cuda:0'))
elif model_name == 'Informer_CNN':
    net = Informer_CNN(enc_in=4, c_out=1, window_size=window_size, factor=5, d_model=64, n_heads=5,
                       e_layers=1, d_layers=0, d_ff=512, dropout=0.5, attn='fully',
                       embed='fixed', freq='h', activation='gelu', output_attention=False, reg_size=32,
                       distil=True, mix=True, device=torch.device('cuda:0'))
elif model_name == 'Informer_TCN':
    net = Informer_TCN(enc_in=4, c_out=1, window_size=window_size, factor=5, d_model=128, n_heads=2,
                       e_layers=2, d_layers=1, d_ff=512, dropout=0.0, attn='prob',
                       embed='fixed', freq='h', activation='gelu', output_attention=False, reg_size=32, num_levels=3,
                       distil=True, mix=True, device=torch.device('cuda:0'))

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(net.parameters(), lr=lr)

trainer = Trainer(model=net, criterion=criterion, train_dataloader=training_dataloader, verbose=True,
                  saving_path=output_dir, val_dataloader=val_dataloader,
                  test_dataloader=None)
trainer.train(epochs=epochs, optimizer=optimizer)

# test
data_name = ['PL21']
input_path = '/public/home/yuwen_hilbert/Program/CALCE_battery/battery_data/data_curve'
TestDataset = CALCEModelDataset(battery_ids=data_name, input_path=input_path, state='test', window_size=window_size,
                                sample_rate=sample_rate, period_rate=1, stride=stride, fully_curve=False, norm=False)
input_features, targets = TestDataset.moving_window()

tester = Tester(model=net, test_data=[input_features, targets], criterion=criterion,input_path=output_dir,
                saving_path=output_dir+fr'/PL21(shallow_model)',
                mode='best', best_epoch=59)
tester.test()

data_name = ['PL23']
input_path = '/public/home/yuwen_hilbert/Program/CALCE_battery/battery_data/data_curve'
TestDataset = CALCEModelDataset(battery_ids=data_name, input_path=input_path, state='test', window_size=window_size,
                                sample_rate=sample_rate, period_rate=1, stride=stride, fully_curve=False, norm=False)
input_features, targets = TestDataset.moving_window()

tester = Tester(model=net, test_data=[input_features, targets], criterion=criterion,input_path=output_dir,
                saving_path=output_dir+fr'/PL23(shallow_model)',
                mode='best', best_epoch=59)
tester.test()