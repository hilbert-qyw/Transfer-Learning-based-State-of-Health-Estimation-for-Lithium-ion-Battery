# -*- coding: gbk -*-
# @Time     : 2022/10/8 13:59
# @Author   : Hilbert
# @Software : PyCharm
from warnings import simplefilter

simplefilter(action='ignore', category=Warning)
simplefilter(action='ignore', category=FutureWarning)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn import manifold
import pickle
import matplotlib as mpl


class CALCEModelDataset(object):
    """
        feed dataset (samples) to model
    """

    def __init__(self,
                 shallow_battery_ids: list,        # the id of battery
                 fully_battery_ids: list,  # the id of battery
                 input_path: str,  # when train model, input path. when test: model, input data
                 state: str,  # training, testing, val
                 window_size: int,  # the length of input sequence
                 sample_rate: int,  # how long sample a point
                 stride: int,       # the stride of moving window
                 fully_curve: bool,     # the label of battery dataset (fully discharge and charge?)
                 norm: bool,  # is normalization
                 period_rate: int,  # the sampling rate of period
                 ):
        self.shallow_battery_ids = shallow_battery_ids
        self.fully_battery_ids = fully_battery_ids
        self.input_path = input_path
        self.state = state
        self.window_size = window_size
        self.sample_rate = sample_rate
        self.stride = stride
        self.period_rate = period_rate
        self.fully_curve = fully_curve
        self.norm = norm
        self.shallow_data, self.shallow_label = self.load_shallow_data()
        self.fully_data, self.fully_label = self.load_fully_data()
        self.font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 32}

    def load_shallow_data(self):
        battery_label = []
        whole_cycles = []
        for label, name in enumerate(self.shallow_battery_ids):
            print("=========Load data===============")
            with open(os.path.join(self.input_path, name + '_charge_curve.pk'), 'rb') as f:
                raw_data = pickle.load(f)
            print("Finish!")
            shallow_curve_data, fully_curve_data = raw_data[0], raw_data[1]
            for period, shallow_charge_data in shallow_curve_data.items():
                cycle_data = shallow_charge_data[-1]
                cycle_data['Calibrate_capacity'] = (fully_curve_data[period][0]['Capacity'].iloc[-1] - 1.1) / (
                            1.45 - 1.1)
                each_period = self.resample(cycle_data).values
                each_period[:, 1] = each_period[:, 1] - each_period[:, 1][0]
                whole_cycles.append(each_period)
                battery_label.append(label)
        return whole_cycles, battery_label

    def load_fully_data(self):
        whole_cycles = []
        for name in self.fully_battery_ids:
            print("=========Load data===============")
            with open(os.path.join(self.input_path, name + '_charge_curve.pk'), 'rb') as f:
                raw_data = pickle.load(f)
            print("Finish!")
            shallow_curve_data, fully_curve_data = raw_data[0], raw_data[1]

            for period, fully_charge_data in fully_curve_data.items():
                if len(fully_charge_data) > 0:
                    for period_idx in np.arange(0, len(fully_charge_data), self.period_rate):
                        cycle_data = fully_charge_data[period_idx]
                        if cycle_data['Capacity'].iloc[-1] > 1:
                            # constant current curve
                            cycle_data['Calibrate_capacity'] = (cycle_data['Capacity'].iloc[-1] - 1.1) / (1.45 - 1.1)
                            # print(cycle_data['Capacity'].iloc[-1])
                            each_period_capacity = cycle_data['Capacity'].iloc[-1]
                            each_period = cycle_data[(cycle_data['Capacity'] > each_period_capacity * 0.2) &
                                                     (cycle_data['Capacity'] < each_period_capacity * 0.8)]
                            each_period = self.resample(each_period).values
                            each_period[:, 1] = each_period[:, 1] - each_period[:, 1][0]
                            whole_cycles.append(each_period)
                            if len(each_period) > 0:
                                whole_cycles.append(each_period)
                            else:
                                continue
        return whole_cycles, [2] * len(whole_cycles)

    def resample(self, cycle_data:pd.DataFrame) -> pd.DataFrame:
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

    def moving_window(self, data):
        inputs = []
        outputs = []
        class_label = []
        for period_idx, cycle in enumerate(data):
            sample_num = len(cycle) - (self.window_size - 1)
            if sample_num < 1:
                print("The length of sequence excesses maximum length of original data!")
                continue
                # raise ValueError("The length of sequence excesses maximum length of original data!")
            else:
                for i in np.arange(0, sample_num, self.stride):
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
                    inputs.append(input_data.reshape(-1))
                    # calibrate capacity
                    outputs.append(cycle[0, 2])
                    class_label.append(period_idx)
        return inputs, outputs, class_label

    def tsne_visualization(self):
        self.shallow_inputs, _, _ = self.moving_window(self.shallow_data)
        self.fully_inputs, _, _ = self.moving_window(self.fully_data)
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=10)
        raw_data = np.concatenate((np.array(self.shallow_inputs), np.array(self.fully_inputs)), axis=0)
        tsne_data = tsne.fit_transform(raw_data)
        fig = plt.figure(figsize=(12, 9))
        color = "hsv"
        color_range = 1.2
        cm = plt.get_cmap(color)
        class_label = self.shallow_label + self.fully_label
        col = [cm(float(i) / class_label[-1] / color_range) for i in class_label][:len(raw_data)]
        cmp = mpl.colors.ListedColormap(col)
        ax1 = fig.add_subplot(111)
        col.reverse()
        ax1.scatter(np.flipud(tsne_data[:, 0]), np.flipud(tsne_data[:, 1]), alpha=1, c=col, s=100)
        ax1.tick_params(direction='out', width=2, length=6)

        label_font = {'family': 'Times New Roman',
                      'weight': 'normal',
                      'size': 28}
        ax1.set_xlabel('Dimension 1', fontdict=label_font, labelpad=10)
        ax1.set_ylabel('Dimension 2', fontdict=label_font, labelpad=10)

        for tick in ax1.get_xticklabels() + ax1.get_yticklabels():
            tick.set_family('Times New Roman')
            tick.set_fontsize(24)

        # set linewidth in spines
        positions = ['top', 'bottom', 'right', 'left']
        for position in positions:
            ax1.spines[position].set_linewidth(4)

        fig.colorbar(mpl.cm.ScalarMappable(cmap=cmp), ax=ax1)
        plt.tight_layout()
        plt.show()


    def charging_curve(self):
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111)
        color = "hsv"
        color_range = 1.2
        cm = plt.get_cmap(color)
        class_label = self.shallow_label + self.fully_label
        col = [cm(float(i) / class_label[-1] / color_range) for i in class_label]
        data = self.shallow_data + self.fully_data
        for idx, curve in enumerate(data):
            ax.plot(curve[:, 0], curve[:, 1], c=col[idx])

        ax.set_xlabel('Voltage (V)', fontdict=self.font)
        ax.set_ylabel('Capacity (Ah)', fontdict=self.font)
        plt.show()

    def window_curve(self):
        self.shallow_inputs, _, _ = self.moving_window(self.shallow_data)
        self.fully_inputs, _, _ = self.moving_window(self.fully_data)
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111)
        fig1 = plt.figure(figsize=(12, 9))
        ax1 = fig1.add_subplot(111)
        color = "hsv"
        color_range = 0.8
        cm = plt.get_cmap(color)
        class_label = [0] * len(self.shallow_inputs) + [1] * len(self.fully_inputs)
        col = [cm(float(i) / len(np.unique(class_label)) / color_range) for i in class_label]
        data = self.shallow_inputs + self.fully_inputs
        data = [curve.reshape(-1, 4) for curve in data]
        for idx, curve in enumerate(data):
            # if (max(curve[:, 2]) > 1):
            #     continue
            # else:
            ax.plot(curve[:, 2], c=col[idx])
            ax1.plot(curve[:, 3], c=col[idx])

        ax.set_ylabel('delta_voltage', fontdict=self.font)
        ax1.set_ylabel('dQ/dV', fontdict=self.font)
        plt.show()

if __name__ == '__main__':
    window_size = 320
    sample_rate = 1
    stride = 2
    period_rate = 20
    shallow_data_name = ['PL21', 'PL23']
    fully_data_name = ['PL11']
    input_path = '/public/home/yuwen_hilbert/Program/CALCE_battery/battery_data/data_curve'
    TrainingDataset = CALCEModelDataset(shallow_battery_ids=shallow_data_name, fully_battery_ids=fully_data_name,
                                      input_path=input_path, state='training',
                                      window_size=window_size,
                                      sample_rate=sample_rate, stride=stride, fully_curve=True, norm=False,
                                      period_rate=period_rate)
    # TrainingDataset.tsne_visualization()
    # TrainingDataset.charging_curve()
    TrainingDataset.window_curve()