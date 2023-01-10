# -*- coding: gbk -*-
# @Time     : 2022/10/14 19:31
# @Author   : Hilbert
# @Software : PyCharm
from warnings import simplefilter

simplefilter(action='ignore', category=Warning)
simplefilter(action='ignore', category=FutureWarning)

from utils import *
from battery_data.CALCE_data_process.lib import *
from model_lib import *
from informer import *

# parameters
lr = 6.10976583060765e-05
dropout = 0.5
d_model = 64
n_head = 4
encoder_layer = 1
cnn_kernel_size = 3
reg_size = 32

model_name = 'CNN1D'
window_size = 160
sample_rate = 2
stride = 2
epochs = 300
sensor_num = 4
batch_size = 16

# last_output_dir = '/public/home/yuwen_hilbert/Program/CALCE_battery/model/result/fully_charge/Informer_CNN/' \
#                   'batch_size=128,lr=6.10976583060765e-05,dropout=0.5,window_size=160,sample_rate=2,d_model=64,' \
#                   'n_head=4,encoder_layer=1,cnn_kernel_size=3,reg_size=32,stride=2,period_rate=3'
last_output_dir = fr'/public/home/yuwen_hilbert/Program/CALCE_battery/model/result/shallow_charge/' \
                  'CNN1D/time_0.00038255850268058_window_size_160_sample_rate_2_' \
                  'stride_1_epochs_150_lr_0.0001_batchsize_16/'

net = CNN1D(window_size=window_size, sensor_num=sensor_num, kernel_size=3, linear_hidden_size_1=128,
                linear_hidden_size_2=64)

# net = SourceTransferInformer_CNN(enc_in=sensor_num, c_out=1, window_size=window_size,
#                                  transfer_fcn_size=64, factor=5,
#                                  d_model=d_model, n_heads=n_head,
#                                  e_layers=encoder_layer, d_layers=encoder_layer - 1, d_ff=512,
#                                  dropout=dropout,
#                                  attn='fully', embed='fixed', freq='h', activation='gelu', output_attention=False,
#                                  cnn_kernel_size=cnn_kernel_size, reg_size=reg_size,
#                                  distil=True, mix=True, device=torch.device('cuda:0'))

# pretrained_dict = torch.load(os.path.join(last_output_dir + '/residual_shallow_model', f'best_model_227.pt'))

pretrained_dict = torch.load(os.path.join(last_output_dir, f'last_model.pt'))
net_dict = net.state_dict()
net_dict.update(pretrained_dict)
net.load_state_dict(net_dict)

input_path = '/public/home/yuwen_hilbert/Program/CALCE_battery/battery_data/data_curve'
data_name = ['PL21']
TrainingDataset = CALCEModelDataset(battery_ids=data_name, input_path=input_path, state='training',
                                    window_size=window_size, sample_rate=sample_rate,
                                    fully_curve=False, norm=False, stride=1, period_rate=1)
training_dataloader = DataLoader(TrainingDataset, batch_size=1, shuffle=False)

transfer_list = []
for input, tar in tqdm(training_dataloader):
    transfer_input = net.transfer_input(input).squeeze().detach().numpy()

    transfer_list.append(transfer_input)
