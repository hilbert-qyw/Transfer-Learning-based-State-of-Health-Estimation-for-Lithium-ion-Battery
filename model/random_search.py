# coding = utf-8
"""
作者   : Hilbert
时间   :2022/7/27 14:48
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
import random
from scipy.stats import loguniform
import argparse
from informer import *

parser = argparse.ArgumentParser(description='Random search')

parser.add_argument('--name', type=str, default='CNN1D')
args = parser.parse_args()

# parameters
epochs = 200
model_name = args.name
sensor_num = 4

# 每次实验，Ray Tune会随机采样超参数组合，并行训练模型，找到最优参数组合
if model_name == 'CNN1D':
    config = {
        # 自定义采样方法
        'window_size': random.choice([96, 128, 160, 192, 224]),
        'sample_rate': random.choice([1, 2, 3]),
        'kernel_size': random.choice([2, 3, 5]),
        'reg_1': 2 ** np.random.randint(5, 8),
        'reg_2': 2 ** np.random.randint(4, 7),
        # 'drop_r': random.choice(list(np.arange(0.2, 0.6, 0.1))),
        'lr': loguniform.rvs(5e-5, 5e-3, size=1)[0],
        'batch_size': 2 ** np.random.randint(6, 10),
    }
    print(config)

    output_dir = f"/public/home/yuwen_hilbert/Program/CALCE_battery/model/result/fully_charge/{model_name}/" \
                 f"batch_size={config['batch_size']},sensor_num={sensor_num}," \
                 f"lr={config['lr']},reg_1={config['reg_1']}," \
                 f"reg_2={config['reg_2']},kernel_size={config['kernel_size']}," \
                 f"sample_rate={config['sample_rate']},window_size={config['window_size']}"
    mkdir(output_dir)
elif model_name == 'CNN_attention':
    config = {
        # 自定义采样方法
        'window_size': random.choice([96, 128, 160, 192, 224]),
        'sample_rate': random.choice([1, 2, 3]),
        'kernel_size': random.choice([2, 3, 5]),
        'reg_1': 2 ** np.random.randint(5, 8),
        'reg_2': 2 ** np.random.randint(4, 7),
        # 'drop_r': random.choice(list(np.arange(0.2, 0.6, 0.1))),
        'lr': loguniform.rvs(5e-5, 5e-3, size=1)[0],
        'batch_size': 2 ** np.random.randint(6, 10),
    }
    print(config)

    output_dir = f"/public/home/yuwen_hilbert/Program/CALCE_battery/model/result/fully_charge/{model_name}/" \
                 f"batch_size={config['batch_size']},sensor_num={sensor_num}," \
                 f"lr={config['lr']},reg_1={config['reg_1']}," \
                 f"reg_2={config['reg_2']},kernel_size={config['kernel_size']}," \
                 f"sample_rate={config['sample_rate']},window_size={config['window_size']}"
    mkdir(output_dir)

elif model_name == 'Informer':
    config = {
        # 自定义采样方法
        'window_size': random.choice([160, 192, 224]),
        'sample_rate': random.choice([1, 2, 3]),
        'd_model': random.choice([256, 512, 1024]),
        'n_head': random.choice([2, 4, 8]),
        'encoder_layer': random.choice([2, 3, 4]),
        'lstm_size': 2 ** np.random.randint(5, 8),
        'lstm_layer': random.choice([2, 3, 4]),
        'reg_size': 2 ** np.random.randint(4, 7),
        'lr': loguniform.rvs(5e-5, 1e-3, size=1)[0],
        'batch_size': 2 ** np.random.randint(6, 10),
    }
    print(config)

    output_dir = f"/public/home/yuwen_hilbert/Program/CALCE_battery/model/result/fully_charge/{model_name}/" \
                 f"batch_size={config['batch_size']},lr={config['lr']},window_size={config['window_size']}," \
                 f"sample_rate={config['sample_rate']},d_model={config['d_model']},n_head={config['n_head']}," \
                 f"encoder_layer={config['encoder_layer']},lstm_size={config['lstm_size']}," \
                 f"lstm_layer={config['lstm_layer']},reg_size={config['reg_size']}"
    mkdir(output_dir)

elif model_name == 'Informer_CNN':
    config = {
        # 自定义采样方法
        'window_size': random.choice([160, 192, 224]),
        'sample_rate': random.choice([2, 3]),
        'd_model': random.choice([64, 128, 256]),
        'n_head': random.choice([1, 2, 4]),
        'encoder_layer': random.choice([1, 2]),
        'cnn_kernel_size': random.choice([2, 3, 5]),
        'reg_size': 2 ** np.random.randint(4, 7),
        'lr': loguniform.rvs(5e-5, 1e-3, size=1)[0],
        'batch_size': 2 ** np.random.randint(6, 10),
        'dropout': random.choice(np.linspace(0.1, 0.5, num=5).tolist()),
    }
    print(config)

    output_dir = f"/public/home/yuwen_hilbert/Program/CALCE_battery/model/result/fully_charge/{model_name}/" \
                 f"batch_size={config['batch_size']},lr={config['lr']}," \
                 f"dropout={config['dropout']},window_size={config['window_size']}," \
                 f"sample_rate={config['sample_rate']},d_model={config['d_model']},n_head={config['n_head']}," \
                 f"encoder_layer={config['encoder_layer']},cnn_kernel_size={config['cnn_kernel_size']}," \
                 f"reg_size={config['reg_size']}"
    mkdir(output_dir)

data_name = ['PL21']
input_path = '/public/home/yuwen_hilbert/Program/CALCE_battery/battery_data/data_curve'
TrainingDataset = ModelDataset(battery_ids=data_name, input_path=input_path, state='training',
                               window_size=config['window_size'], sample_rate=config['sample_rate'],
                               fully_curve=True, norm=False)
training_dataloader = DataLoader(TrainingDataset, batch_size=config['batch_size'], shuffle=True)

data_name = ['PL23']
ValDataset = ModelDataset(battery_ids=data_name, input_path=input_path, state='training',
                          window_size=config['window_size'], sample_rate=config['sample_rate'],
                          fully_curve=True, norm=False)
val_dataloader = DataLoader(ValDataset, batch_size=config['batch_size'], shuffle=False)

if model_name == 'CNN1D':
    net = CNN1D(window_size=config['window_size'], sensor_num=sensor_num, kernel_size=config['kernel_size'],
                linear_hidden_size_1=config['reg_1'], linear_hidden_size_2=config['reg_2'])
elif model_name == 'LSTM':
    net = LSTM(input_timestep=config['window_size'], sensor_num=sensor_num)
elif model_name == 'CNN_attention':
    net = CNN_attention(window_size=config['window_size'], sensor_num=sensor_num, kernel_size=config['kernel_size'],
                        linear_hidden_size_1=config['reg_1'], linear_hidden_size_2=config['reg_2'])
elif model_name == 'Informer':
    net = Informer(enc_in=sensor_num, c_out=1, factor=5, d_model=config['d_model'], n_heads=config['n_head'],
                   e_layers=config['encoder_layer'], d_layers=config['encoder_layer']-1, d_ff=512, dropout=0.0,
                   attn='prob', embed='fixed', freq='h', activation='gelu', output_attention=False,
                   lstm_hidden_size=config['lstm_size'], lstm_hidden_layers=config['lstm_layer'], reg_size=config['reg_size'],
                   distil=True, mix=True, device=torch.device('cuda:0'))
elif model_name == 'Informer_CNN':
    net = Informer_CNN(enc_in=sensor_num, c_out=1, window_size=config['window_size'], factor=5,
                       d_model=config['d_model'], n_heads=config['n_head'],
                       e_layers=config['encoder_layer'], d_layers=config['encoder_layer']-1, d_ff=512,
                       dropout=config['dropout'],
                       attn='fully', embed='fixed', freq='h', activation='gelu', output_attention=False,
                       cnn_kernel_size=config['cnn_kernel_size'], reg_size=config['reg_size'],
                       distil=True, mix=True, device=torch.device('cuda:0'))

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(net.parameters(), lr=config['lr'])

trainer = Trainer(model=net, criterion=criterion, train_dataloader=training_dataloader, verbose=True,
                  saving_path=output_dir, val_dataloader=val_dataloader,
                  test_dataloader=None)
trainer.train(epochs=epochs, optimizer=optimizer)

# test
data_name = ['PL21']
input_path = '/public/home/yuwen_hilbert/Program/CALCE_battery/battery_data/data_curve'
TestDataset = ModelDataset(battery_ids=data_name, input_path=input_path, state='test',
                           window_size=config['window_size'], sample_rate=config['sample_rate'],
                           fully_curve=False, norm=False)
input_features, targets = TestDataset.moving_window()

tester = Tester(model=net, test_data=[input_features, targets], criterion=criterion, saving_path=output_dir,
                mode='last', best_epoch=1)
tester.test()