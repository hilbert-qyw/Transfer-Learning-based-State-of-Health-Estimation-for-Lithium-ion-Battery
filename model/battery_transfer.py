# coding = utf-8
"""
作者   : Hilbert
时间   :2022/8/4 14:46
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
import argparse
from scipy.stats import loguniform


# parameters
lr = 6.10976583060765e-05
dropout = 0.5
d_model = 64
n_head = 4
encoder_layer = 1
cnn_kernel_size = 3
reg_size = 32

parser = argparse.ArgumentParser(description='Random search')
parser.add_argument('--name', type=str, default='CNN1D')
args = parser.parse_args()
model_name = args.name
window_size = 160
sample_rate = 2
stride = 2
epochs = 300
sensor_num = 4

last_output_dir = '/public/home/yuwen_hilbert/Program/CALCE_battery/model/result/fully_charge/Informer_CNN/' \
                  'batch_size=128,lr=6.10976583060765e-05,dropout=0.5,window_size=160,sample_rate=2,d_model=64,' \
                  'n_head=4,encoder_layer=1,cnn_kernel_size=3,reg_size=32,stride=2,period_rate=3'

if model_name == 'CNN1D':
    net = SourceTransferCNN1D(window_size=window_size, sensor_num=sensor_num,
                              kernel_size=3,
                              linear_hidden_size_1=128, linear_hidden_size_2=64)
elif model_name == 'Informer_CNN':
    net = SourceTransferInformer_CNN(enc_in=sensor_num, c_out=1, window_size=window_size, factor=5,
                                     d_model=d_model, n_heads=n_head,
                                     e_layers=encoder_layer, d_layers=encoder_layer - 1, d_ff=512,
                                     dropout=dropout,
                                     attn='fully', embed='fixed', freq='h', activation='gelu', output_attention=False,
                                     cnn_kernel_size=cnn_kernel_size, reg_size=reg_size,
                                     distil=True, mix=True, device=torch.device('cuda:0'))

# transfer learning parameters
config = {'batch_size': 2 ** np.random.randint(0, 6),
          'lr': loguniform.rvs(1e-5, 1e-3, size=1)[0],
          'last_model': np.random.choice([True, False]),
          'modify_regression': np.random.choice([True, False]),
          'l2_regularization': np.random.choice([True, False]),
          'weight_decay': loguniform.rvs(1e-5, 1e-3, size=1)[0]}
print("========Parameters setting==========")
print(config)
print("====================================")

# 导入部分参数
net_dict = net.state_dict()
if config['last_model']:
    pretrained_dict = torch.load(os.path.join(last_output_dir + '/fully_model', f'last_model.pt'))
else:
    pretrained_dict = torch.load(os.path.join(last_output_dir + '/fully_model', f'best_model_42.pt'))

# pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'cnn1d' in k}
net_dict.update(pretrained_dict)
net.load_state_dict(net_dict)

for p in net.parameters():
    p.requires_grad = False
# train the parameters of fully connected layers
for p in net.source_transfer.parameters():
    p.requires_grad = True

if config['modify_regression']:
    for p in net.regression.parameters():
        p.requires_grad = True

if config['l2_regularization']:
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=config['lr'],
                                  weight_decay=config['weight_decay'])
else:
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=config['lr'])

criterion = nn.MSELoss()
#

input_path = '/public/home/yuwen_hilbert/Program/CALCE_battery/battery_data/data_curve'
data_name = ['PL21']
TrainingDataset = CALCEModelDataset(battery_ids=data_name, input_path=input_path, state='training',
                                    window_size=window_size, sample_rate=sample_rate,
                                    fully_curve=False, norm=False, stride=1, period_rate=1)
training_dataloader = DataLoader(TrainingDataset, batch_size=config['batch_size'], shuffle=True)

data_name = ['PL23']
ValDataset = CALCEModelDataset(battery_ids=data_name, input_path=input_path, state='training',
                               window_size=window_size, sample_rate=sample_rate,
                               fully_curve=False, norm=False, stride=1, period_rate=1)
val_dataloader = DataLoader(ValDataset, batch_size=config['batch_size'], shuffle=False)

trainer = Trainer(model=net, criterion=criterion, train_dataloader=training_dataloader, verbose=True,
                  saving_path=last_output_dir + '/shallow_model', val_dataloader=val_dataloader,
                  test_dataloader=None)
trainer.train(epochs=epochs, optimizer=optimizer)

# test
data_name = ['PL23']
TestDataset = CALCEModelDataset(battery_ids=data_name, input_path=input_path,
                                state='test', window_size=window_size,
                                sample_rate=sample_rate, fully_curve=False,
                                norm=False, stride=1, period_rate=1)
input_features, targets = TestDataset.moving_window()

tester = Tester(model=net, test_data=[input_features, targets], criterion=criterion,
                input_path=last_output_dir + '/shallow_model',
                saving_path=last_output_dir + '/PL23(shallow_model)', mode='last')
tester.test()