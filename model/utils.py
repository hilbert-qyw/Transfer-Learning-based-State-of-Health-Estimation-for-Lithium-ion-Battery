# coding = utf-8
"""
作者   : Hilbert
时间   :2022/7/23 14:29
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


from battery_data.CALCE_data_process.lib import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import copy
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.utils.data import TensorDataset


class CALCEModelDataset(Dataset):
    """
        feed dataset (samples) to model
    """

    def __init__(self,
                 battery_ids: list,        # the id of battery
                 input_path: str,  # when train model, input path. when test: model, input data
                 state: str,  # training, testing, val
                 window_size: int,  # the length of input sequence
                 sample_rate: int,  # how long sample a point
                 stride: int,  # the stride of moving window
                 fully_curve: bool,     # the label of battery dataset (fully discharge and charge?)
                 norm: bool,  # is normalization
                 period_rate: int,  # the sampling rate of period
                 ):
        self.battery_ids = battery_ids
        self.input_path = input_path
        self.state = state
        self.window_size = window_size
        self.sample_rate = sample_rate
        self.fully_curve = fully_curve
        self.norm = norm
        self.stride = stride
        self.period_rate = period_rate
        self.data = self.load_data()
        self.inputs, self.outputs = self.moving_window()

    def load_data(self):
        whole_cycles = []
        len_cycles = []
        for name in self.battery_ids:
            print("=========Load data===============")
            with open(os.path.join(self.input_path, name + '_charge_curve.pk'), 'rb') as f:
                raw_data = pickle.load(f)
            print("Finish!")
            shallow_curve_data, fully_curve_data = raw_data[0], raw_data[1]

            if self.fully_curve:
                for period, fully_charge_data in fully_curve_data.items():
                    if len(fully_charge_data) > 0:
                        for cycle_data in fully_charge_data:
                            if cycle_data['Capacity'].iloc[-1] > 1:
                                # constant current curve
                                cycle_data['Calibrate_capacity'] = (cycle_data['Capacity'].iloc[-1] - 1.1) / (1.45 - 1.1)
                                # print(cycle_data['Capacity'].iloc[-1])
                                cycle_data = self.resample(cycle_data)
                                if len(cycle_data) > 0:
                                    whole_cycles.append(cycle_data.values)
                                else:
                                    continue
                                len_cycles.append(len(cycle_data))

            else:
                for period, shallow_charge_data in shallow_curve_data.items():
                    cycle_data = shallow_charge_data[-1]
                    cycle_data['Calibrate_capacity'] = (fully_curve_data[period][0]['Capacity'].iloc[-1] - 1.1) / (1.45 - 1.1)
                    cycle_data = self.resample(cycle_data)
                    if len(cycle_data) > 0:
                        whole_cycles.append(cycle_data.values)
                    else:
                        continue
                    len_cycles.append(len(cycle_data))

        if max(len_cycles) - min(len_cycles) > 500:
            raise ValueError('请检查充电区间划分！')

        return whole_cycles

    def resample(self, cycle_data:pd.DataFrame):
        # uniform sampling rate
        cycle_data = cycle_data[cycle_data['Current_Amp'] > 0.7]
        time = pd.to_datetime(cycle_data['Date_Time'] - 719529, unit='d').round('us')
        cycle_data = cycle_data.set_index(time)
        sample = cycle_data[['Voltage_Volt', 'Capacity', 'Calibrate_capacity']]
        sample = sample.resample('10s').mean()
        resample = sample.interpolate(method='linear')
        time_diff = time.diff().iloc[1:].apply(lambda x: x.total_seconds())
        if max(time_diff) > 100:
            resample = []

        return resample

    def moving_window(self):
        inputs = []
        outputs = []
        if self.state == 'training':
            for cycle in self.data[::self.period_rate]:
                sample_num = len(cycle) - (self.window_size - 1) * self.sample_rate
                if sample_num < 1:
                    raise ValueError("The length of sequence excesses maximum length of original data!")
                else:
                    for i in np.arange(0, sample_num, self.stride):
                        position = np.linspace(start=i, stop=i+self.window_size*self.sample_rate, num=self.window_size,
                                               endpoint=False, dtype=int)
                        # voltage and increased capacity
                        input_data = cycle[position, :2]
                        input_data[:, 1] = input_data[:, 1] - input_data[0, 1]
                        # voltage, delta_Q, delta_v, delta_Q/delta_V
                        delta_V = input_data[:, 0] - input_data[0, 0]
                        # delta_Q_V = np.append([0], np.diff(input_data[:, 1]) / np.diff(delta_V))
                        delta_Q_V = input_data[:, 1] / delta_V
                        input_data = np.concatenate([input_data, delta_V[:, np.newaxis], delta_Q_V[:, np.newaxis]], axis=1)
                        input_data[np.isnan(input_data)] = 0
                        input_data[np.isinf(input_data)] = 0
                        inputs.append(input_data)
                        # calibrate capacity
                        outputs.append(cycle[0, 2])
        elif self.state == 'val':
            for cycle in self.data[::self.period_rate]:
                sample_num = len(cycle) - (self.window_size - 1) * self.sample_rate
                if sample_num < 1:
                    raise ValueError("The length of sequence excesses maximum length of original data!")
                else:
                    for i in np.arange(0, sample_num, self.stride):
                        position = np.linspace(start=i, stop=i+self.window_size*self.sample_rate, num=self.window_size,
                                               endpoint=False, dtype=int)
                        # voltage and increased capacity
                        input_data = cycle[position, :2]
                        input_data[:, 1] = input_data[:, 1] - input_data[0, 1]
                        # voltage, delta_Q, delta_v, delta_Q/delta_V
                        delta_V = input_data[:, 0] - input_data[0, 0]
                        # delta_Q_V = np.append([0], np.diff(input_data[:, 1]) / np.diff(delta_V))
                        delta_Q_V = input_data[:, 1] / delta_V
                        input_data = np.concatenate([input_data, delta_V[:, np.newaxis], delta_Q_V[:, np.newaxis]], axis=1)
                        input_data[np.isnan(input_data)] = 0
                        input_data[np.isinf(input_data)] = 0
                        inputs.append(input_data)
                        # calibrate capacity
                        outputs.append(cycle[0, 2])

        else:
            for cycle in self.data:
                cycle_input = []
                cycle_output = []
                sample_num = len(cycle) - (self.window_size - 1) * self.sample_rate
                if sample_num < 1:
                    raise ValueError("The length of sequence excesses maximum length of original data!")
                else:
                    for i in np.arange(0, sample_num, self.stride):
                        position = np.linspace(start=i, stop=i+self.window_size*self.sample_rate, num=self.window_size,
                                               endpoint=False, dtype=int)
                        # voltage and increased capacity
                        input_data = cycle[position, :2]
                        input_data[:, 1] = input_data[:, 1] - input_data[0, 1]
                        # voltage, delta_Q, delta_v, delta_Q/delta_V
                        delta_V = input_data[:, 0] - input_data[0, 0]
                        # delta_Q_V = np.append([0], np.diff(input_data[:, 1]) / np.diff(delta_V))
                        delta_Q_V = input_data[:, 1] / delta_V
                        input_data = np.concatenate([input_data, delta_V[:, np.newaxis], delta_Q_V[:, np.newaxis]], axis=1)
                        input_data[np.isnan(input_data)] = 0
                        input_data[np.isinf(input_data)] = 0
                        cycle_input.append(input_data)
                        # calibrate capacity
                        cycle_output.append(cycle[0, 2])
                inputs.append(cycle_input)
                outputs.append(cycle_output)

        return inputs, outputs

    def __getitem__(self, index):
        inputs = self.inputs[index].astype(np.float32)
        target = self.outputs[index].astype(np.float32)
        return inputs, target

    def __len__(self):
        return len(self.inputs)


class SNLModelDataset(Dataset):
    """
        feed dataset (samples) to model
    """

    def __init__(self,
                 battery_ids: list,        # the id of battery
                 input_path: str,  # when train model, input path. when test: model, input data
                 state: str,  # training, testing, val
                 window_size: int,  # the length of input sequence
                 sample_rate: int,  # how long sample a point
                 fully_curve: bool,     # the label of battery dataset (fully discharge and charge?)
                 norm: bool,  # is normalization
                 ):
        self.battery_ids = battery_ids
        self.input_path = input_path
        self.state = state
        self.window_size = window_size
        self.sample_rate = sample_rate
        self.fully_curve = fully_curve
        self.norm = norm
        self.data = self.load_data()
        self.inputs, self.outputs = self.moving_window()

    def load_data(self):
        for name in self.battery_ids:
            print("=========Load data===============")
            with open(os.path.join(self.input_path, name + '_charge_curve.pk'), 'rb') as f:
                raw_data = pickle.load(f)
            print("Finish!")
            shallow_curve_data, fully_curve_data = raw_data[0], raw_data[1]
            whole_cycles = []
            if self.fully_curve:
                for each_period in fully_curve_data:
                    each_period['Calibrate_capacity'] = (each_period['Calibrate_capacity'] - 0.9) / (1.1 - 0.9)
                    whole_cycles.append(self.resample(each_period).values)
            else:
                for each_period in shallow_curve_data:
                    each_period['Calibrate_capacity'] = (each_period['Calibrate_capacity'] - 0.9) / (1.1 - 0.9)
                    whole_cycles.append(self.resample(each_period).values)
        return whole_cycles

    def resample(self, cycle_data:pd.DataFrame) -> pd.DataFrame:
        # uniform sampling rate
        time = pd.to_datetime('2020/1/1 0:00:00')
        cycle_data['Test_Time (s)'] = cycle_data['Test_Time (s)'].apply(lambda x: time+pd.Timedelta(x, unit='seconds'))
        cycle_data = cycle_data.set_index(cycle_data['Test_Time (s)'])
        sample = cycle_data[['Voltage (V)', 'Charge_Capacity (Ah)', 'Calibrate_capacity']]
        sample = sample.resample('5s').mean()
        resample = sample.interpolate(method='linear')
        # time_diff = time.diff().iloc[1:].apply(lambda x: x.total_seconds())
        # if max(time_diff) > 100:
        #     resample = []
        return resample

    def moving_window(self):
        inputs = []
        outputs = []
        if self.state == 'training':
            for cycle in self.data:
                sample_num = len(cycle) - (self.window_size - 1) * self.sample_rate
                if sample_num < 1:
                    raise ValueError("The length of sequence excesses maximum length of original data!")
                else:
                    for i in range(sample_num):
                        position = np.linspace(start=i, stop=i+self.window_size*self.sample_rate, num=self.window_size,
                                               endpoint=False, dtype=int)
                        # voltage and increased capacity
                        input_data = cycle[position, :2]
                        input_data[:, 1] = input_data[:, 1] - input_data[0, 1]
                        # voltage, delta_Q, delta_v, delta_Q/delta_V
                        delta_V = input_data[:, 0] - input_data[0, 0]
                        delta_Q_V = np.append([0], np.diff(input_data[:, 1]) / np.diff(delta_V))
                        input_data = np.concatenate([input_data, delta_V[:, np.newaxis], delta_Q_V[:, np.newaxis]], axis=1)
                        input_data[np.isnan(input_data)] = 0
                        input_data[np.isinf(input_data)] = 0
                        inputs.append(input_data)
                        # calibrate capacity
                        outputs.append(cycle[0, 2])

        else:
            inputs = []
            outputs = []
            for cycle in self.data:
                cycle_input = []
                cycle_output = []
                sample_num = len(cycle) - (self.window_size - 1) * self.sample_rate
                if sample_num < 1:
                    raise ValueError("The length of sequence excesses maximum length of original data!")
                else:
                    for i in range(sample_num):
                        position = np.linspace(start=i, stop=i+self.window_size*self.sample_rate, num=self.window_size,
                                               endpoint=False, dtype=int)
                        # voltage and increased capacity
                        input_data = cycle[position, :2]
                        input_data[:, 1] = input_data[:, 1] - input_data[0, 1]
                        # voltage, delta_Q, delta_v, delta_Q/delta_V
                        delta_V = input_data[:, 0] - input_data[0, 0]
                        delta_Q_V = np.append([0], np.diff(input_data[:, 1]) / np.diff(delta_V))
                        input_data = np.concatenate([input_data, delta_V[:, np.newaxis], delta_Q_V[:, np.newaxis]], axis=1)
                        input_data[np.isnan(input_data)] = 0
                        input_data[np.isinf(input_data)] = 0
                        cycle_input.append(input_data)
                        # calibrate capacity
                        cycle_output.append(cycle[0, 2])
                inputs.append(cycle_input)
                outputs.append(cycle_output)

        return inputs, outputs

    def __getitem__(self, index):
        inputs = self.inputs[index].astype(np.float32)
        target = self.outputs[index].astype(np.float32)
        return inputs, target

    def __len__(self):
        return len(self.inputs)


class Trainer(object):
    def __init__(
            self,
            model: nn.Module,
            criterion=None,
            train_dataloader=None,
            *,
            scheduler=None,
            device='cpu',
            verbose=True,
            saving_path= rootPath + '/results',
            val_dataloader=None,
            test_dataloader=None,

    ) -> None:
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.device = device
        self.scheduler = scheduler
        self.verbose = verbose
        self.saving_path = saving_path
        mkdir(self.saving_path)

        if val_dataloader is not None:
            self.val_dataloader = val_dataloader

        if test_dataloader is not None:
            self.test_dataloader = test_dataloader

    def train(self, epochs=50, optimizer=None, ):
        if optimizer is not None:
            self.optimizer = optimizer

        print("=> Beginning training")

        train_loss = []
        val_loss = []
        best_loss = None
        self.model.train()

        for epoch in range(epochs):
            train_batch_loss = []
            print('========Epoch(train)-%d========' % epoch)
            # with tqdm(total=self.train_dataloader.__len__()) as pbar:
            #     pbar.set_description('Processing:')
            for input, tar in tqdm(self.train_dataloader, desc='Epoch(train)-%d' % epoch):
                # for input, tar in self.train_dataloader:
                if len(input[torch.isnan(input)]):
                    print(input[torch.isnan(input)])
                self.optimizer.zero_grad()
                pre = self.model(input)
                loss = self.criterion(pre, tar)

                loss.backward()
                self.optimizer.step()

                # 检查梯度是否正常
                # for name, weight in self.model.named_parameters():
                #     if weight.requires_grad:
                #         print(f"{name} weight.grad:", weight.grad.mean(), weight.grad.min(), weight.grad.max())

                train_batch_loss.append(loss.item())

                    # pbar.update(1)

            if self.scheduler is not None:
                self.scheduler.step()

            train_epoch_loss = sum(train_batch_loss) / len(train_batch_loss)

            val_epoch_loss, val_err, _ = self.eval(self.val_dataloader)

            if self.verbose:
                print(f'Epoch:{epoch:3d}\nTraining Loss:{train_epoch_loss:.4f}\tValidation Loss:{val_epoch_loss:.4f}',
                      flush=True)
                print(f'Validation metrics:\nRMSE:{val_err[0]:.4f}\tMAE:{val_err[1]:.4f}'
                      f'\tR2:{val_err[2]:.4f}', flush=True)

            if best_loss == None or val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                self.best_model = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch
                best_err = val_err
                print("successfully save best model!")

            train_loss.append(train_epoch_loss)
            val_loss.append(val_epoch_loss)


        self.history = {'train loss': train_loss, 'val loss': val_loss}
        print("Best performance:")
        print(f'Test metrics:\nRMSE:{best_err[0]:.4f}\tMAE:{best_err[1]:.4f}'
              f'\tR2:{best_err[2]:.4f}\n', flush=True)

        print("=> Saving model to file")

        if not os.path.exists(self.saving_path):
            os.mkdir(self.saving_path)
        torch.save(self.model.state_dict(), os.path.join(self.saving_path, 'last_model.pt'))
        torch.save(self.best_model, os.path.join(self.saving_path, f'best_model.pt'))
        torch.save(self.history, os.path.join(self.saving_path, 'loss_history.pt'))

        self.plot_loss()

        return self.history

    def eval(self, data_loader, save_data=False, save_plot=False, name=None):
        self.model.eval()
        with torch.no_grad():
            y_true = []
            y_predict = []
            cum_loss = []
            for input, tar in data_loader:
                pre = self.model(input)
                loss = self.criterion(pre, tar)

                cum_loss.append(loss.item())

                y_true.append(tar.detach().numpy())
                y_predict.append(pre.detach().numpy())

            val_epoch_loss = sum(cum_loss) / len(cum_loss)

        y_true = np.concatenate(y_true)
        # print(y.shape)
        y_predict = np.concatenate(y_predict)

        mae = mean_absolute_error(y_true, y_predict)
        mse = mean_squared_error(y_true, y_predict)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_predict)
        print(f'Val metrics: \tRMSE:{rmse}, MAE:{mae}, R2:{r2}')

        # print(len(y))

        if save_data:
            np.savetxt(os.path.join(self.saving_path, name + '_label.txt'), y_true)
            np.savetxt(os.path.join(self.saving_path, name + '_predict.txt'), y_predict)
            with open(os.path.join(self.saving_path, name + '_metrics.txt'), 'w') as f:
                print(f'\tRMSE:{rmse}, MAE:{mae}, R2:{r2}', file=f)

        if save_plot:
            plt.figure()
            plt.plot(y_true, 'k', label='target')
            plt.plot(y_predict, 'r', label='predict')
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.saving_path + '/' + name + fr'target_predict.png')
            plt.cla()

            plt.plot(y_true - y_predict)
            plt.ylabel('prediction error')
            plt.tight_layout()
            plt.savefig(self.saving_path + '/' + name + fr'target_predict_error.png')
            plt.clf()

            plt.scatter(y_true, y_predict)
            plt.xlabel('target value')
            plt.ylabel('predicted value')
            plt.tight_layout()
            plt.savefig(self.saving_path + '/' + name + fr'target_predict_scatter.png')

            plt.clf()

        return val_epoch_loss, (rmse, mae, r2), (y_true, y_predict)

    def test(self, mode='last', save_data=False, save_plot=True):
        print("\n=> Evaluating " + mode + " model on test dataset")

        if mode == 'last':
            model = self.last_model
        else:
            model = self.best_model

        self.model.load_state_dict(model)
        test_loss, metrics, y = self.eval(self.test_dataloader, save_data=save_data, save_plot=save_plot, name=mode)

        y_true = y[0] * (1.45 - 1.1) + 1.1
        y_pred = y[1] * (1.45 - 1.1) + 1.1
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        print(f'Test loss:{test_loss}, RMSE:{rmse}, MAE:{mae}, R2:{r2}')

        return rmse


    def plot_loss(self):
        epoch_arr = list(range(len(self.history['train loss'])))

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(epoch_arr, self.history['train loss'], )
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('train loss')

        plt.subplot(2, 1, 2)
        plt.plot(epoch_arr, self.history['val loss'], )
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('val loss')

        savepath = os.path.join(self.saving_path, 'train_loss.png')
        plt.savefig(savepath)
        plt.clf()


class Tester(object):
    def __init__(
            self,
            model: nn.Module,
            test_data=None,         # list [input_features, targets]
            criterion=None,
            device='cpu',
            input_path='./results',
            saving_path='./results',
            mode='best',
            best_epoch=None,
    ) -> None:
        super().__init__()
        self.model = model
        if mode == 'best':
            self.model.load_state_dict(torch.load(os.path.join(input_path, f'best_model.pt')))
        elif mode == 'last':
            self.model.load_state_dict(torch.load(os.path.join(input_path, 'last_model.pt')))
        self.test_data = test_data
        self.device = device
        self.saving_path = saving_path
        mkdir(self.saving_path)
        self.criterion = criterion
        self.font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 32}

    def test(self, mode='last', save_data=False, save_plot=False):
        print("\n=> Evaluating " + mode + " model on test dataset")
        y_mean = []
        y_median = []
        y_true = []
        y_all_true = []
        y_all_pre = []
        for (cycle_input, cycle_output) in zip(self.test_data[0], self.test_data[1]):
            cycle_input, cycle_output = torch.Tensor(cycle_input), torch.Tensor(cycle_output)
            cycle_output = cycle_output.view(-1)
            cycle_dataset = TensorDataset(cycle_input, cycle_output)
            cycle_dataloader = DataLoader(cycle_dataset, shuffle=False)
            test_loss, y = self.eval(cycle_dataloader, save_data=save_data, save_plot=save_plot, name=mode)

            # calculate mean capacity
            y_mean.append(np.mean(y[1]))
            y_median.append(np.median(y[1]))
            y_true.append(y[0][0])

            # all capacity
            y_all_true.append(y[0])
            y_all_pre.append(y[1])

        y_all_true = np.concatenate(y_all_true)
        # print(y.shape)
        y_all_pre = np.concatenate(y_all_pre)

        rmse = np.sqrt(mean_squared_error(y_all_pre, y_all_true))
        mean_rmse = np.sqrt(mean_squared_error(y_mean, y_true))
        median_rmse = np.sqrt(mean_squared_error(y_median, y_true))
        print(f'RMSE:{rmse}'
              f'\nMean RMSE:{mean_rmse}'
              f'\n Median RMSE:{median_rmse}')

        # plot cycle prediction results
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111)
        ax.set_title('Predition capacity faded curve', fontdict=self.font)
        ax.set_xlabel('sample point', fontdict=self.font)
        ax.set_ylabel('capacity', fontdict=self.font)
        ax.plot(y_true, label='True')
        ax.plot(y_mean, label='Mean')
        ax.plot(y_median, label='Median')
        # ax.legend()
        plt.legend(prop=self.font)
        plt.tight_layout()
        plt.savefig(self.saving_path + fr'/capacity prediction.png')

        # plot window prediciton results
        self.plot_curve(title='Window capacity prediction result', xlabel='window number', ylabel='capacity',
                        x_axis=[list(range(len(y_all_true)))]*2, y_axis=[y_all_pre, y_all_true],
                        name='Window_capacity_prediction_result')

    def eval(self, data_loader, save_data=False, save_plot=False, name=None):
        self.model.eval()
        with torch.no_grad():
            y_true = []
            y_predict = []
            cum_loss = []
            for input, tar in data_loader:
                pre = self.model(input)
                loss = self.criterion(pre, tar)

                cum_loss.append(loss.item())

                y_true.append(tar.detach().numpy())
                y_predict.append(pre.detach().numpy())

            val_epoch_loss = sum(cum_loss) / len(cum_loss)

        # print(y.shape)
        # CALCE
        # y_predict = np.concatenate(y_predict) * (1.45 - 1.1) + 1.1
        # y_true = np.concatenate(y_true) * (1.45 - 1.1) + 1.1
        # SNL
        y_predict = np.concatenate(y_predict) * (1.45 - 1.1) + 1.1
        y_true = np.concatenate(y_true) * (1.45 - 1.1) + 1.1

        if save_plot:
            # plot the distribution of prediction result
            fig = plt.figure(figsize=(16, 12))
            ax = fig.add_subplot(111)
            ax.set_title('The distribution of prediction result', fontdict=self.font)
            ax.set_xlabel('Capacity', fontdict=self.font)
            # ax.set_ylabel('Capacity', fontdict=self.font)
            ax.hist(y_predict)
            plt.show()
            # plt.savefig(self.saving_path + fr'/capacity prediction.png')

        return val_epoch_loss, (y_true, y_predict)

    def plot_curve(self, title, xlabel, ylabel, x_axis, y_axis, name):
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111)
        ax.set_title(title, fontdict=self.font)
        ax.set_xlabel(xlabel, fontdict=self.font)
        ax.set_ylabel(ylabel, fontdict=self.font)
        for (x, y) in zip(x_axis, y_axis):
            ax.plot(x, y)
        plt.tight_layout()
        plt.savefig(self.saving_path + fr'/{name}.png')