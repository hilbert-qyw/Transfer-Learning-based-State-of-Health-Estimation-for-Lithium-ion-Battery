# -*- coding: gbk -*-
# @Time     : 2022/9/18 15:55
# @Author   : Hilbert
# @Software : PyCharm
from warnings import simplefilter

simplefilter(action='ignore', category=Warning)
simplefilter(action='ignore', category=FutureWarning)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle


class CALCEVisualization(object):
    """
        feed dataset (samples) to model
    """

    def __init__(self,
                 battery_ids: list,        # the id of battery
                 input_path: str,  # when train model, input path. when test: model, input data
                 state: str,  # training, testing, val
                 window_size: int,  # the length of input sequence
                 sample_rate: int,  # how long sample a point
                 stride: int,       # the stride of moving window
                 fully_curve: bool,     # the label of battery dataset (fully discharge and charge?)
                 norm: bool,  # is normalization
                 period_rate: int,  # the sampling rate of period
                 ):
        self.battery_ids = battery_ids
        self.input_path = input_path
        self.state = state
        self.window_size = window_size
        self.sample_rate = sample_rate
        self.stride = stride
        self.period_rate = period_rate
        self.fully_curve = fully_curve
        self.norm = norm
        self.font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 32}

    def load_data(self, plot_setting=True):
        whole_cycles = []
        len_cycles = []
        capacity = []
        for name in self.battery_ids:
            print("=========Load data===============")
            with open(os.path.join(self.input_path, name + '_charge_curve.pk'), 'rb') as f:
                raw_data = pickle.load(f)
            print("Finish!")
            shallow_curve_data, fully_curve_data = raw_data[0], raw_data[1]

            for period, shallow_charge_data in shallow_curve_data.items():
                cycle_data = shallow_charge_data[-1]
                cycle_data['Calibrate_capacity'] = (fully_curve_data[period][0]['Capacity'].iloc[-1] - 1.1) / (
                            1.45 - 1.1)
                capacity.append(fully_curve_data[period][0]['Capacity'].iloc[-1] )
                cycle_data = self.resample(cycle_data)
                if len(cycle_data) > 0:
                    whole_cycles.append(cycle_data)
                else:
                    continue
                len_cycles.append(len(cycle_data))


        if plot_setting:
            for idx, curve in enumerate(whole_cycles):
                print(idx, curve['Voltage_Volt'].min(), curve['Voltage_Volt'].max())
            # plot the voltage curve in shallow charge
            fig = plt.figure(figsize=(16, 12))
            ax = fig.add_subplot(111)
            for idx, curve in enumerate(whole_cycles):
                ax.plot(curve['Voltage_Volt'].values, label=fr'line_{idx}')
            # ax.set_xlabel('Time', fontdict=self.font)
            ax.set_ylabel('Voltage (V)', fontdict=self.font)
            ax.set_title('Shallow charge voltage', fontdict=self.font)
            ax.legend()
            plt.show()

            # plot the capacity degradation curve in shallow charge
            fig = plt.figure(figsize=(16, 12))
            ax = fig.add_subplot(111)
            ax.plot(capacity)
            # ax.set_xlabel('Time', fontdict=self.font)
            ax.set_ylabel('Capacity (Ah)', fontdict=self.font)
            ax.set_title('Capacity degradation', fontdict=self.font)
            plt.show()

            # plot the voltage vs capacity curve in shallow charge
            # fig = plt.figure(figsize=(16, 12))
            # ax = fig.add_subplot(111)
            # for curve in fully_curve_data:
            #     ax.plot(curve['Voltage (V)'].values, curve['Charge_Capacity (Ah)'].values)
            # ax.set_xlabel('Voltage (V)', fontdict=self.font)
            # ax.set_ylabel('Capacity (Ah)', fontdict=self.font)
            # ax.set_title('Voltage vs. Capacity', fontdict=self.font)
            # plt.show()

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

if __name__ == '__main__':
    input_path = '/public/home/yuwen_hilbert/Program/CALCE_battery/battery_data/data_curve'
    data_name = ['PL23']
    window_size = 160
    sample_rate = 2
    stride = 32
    period_rate = 1

    TrainingDataset = CALCEVisualization(battery_ids=data_name, input_path=input_path, state='training',
                                         window_size=window_size,
                                         sample_rate=sample_rate, stride=stride, fully_curve=True, norm=False,
                                         period_rate=period_rate)
    TrainingDataset.load_data(plot_setting=True)