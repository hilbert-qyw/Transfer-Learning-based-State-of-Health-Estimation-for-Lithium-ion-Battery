# -*- coding: gbk -*-
# @Time     : 2022/9/12 22:00
# @Author   : Hilbert
# @Software : PyCharm
from warnings import simplefilter
simplefilter(action='ignore', category=Warning)
simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle


def mkdir(path):
    """
    mkdir of the path
    :param input: string of the path
    return: boolean
    """
    path = path.strip()
    path = path.rstrip('\\')
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' is created!')
        return True
    else:
        print(path+' already exists!')
        return False


class BatteryDataSegment(object):
    def __init__(self, input_path: str, file_name: str, saving_path: str, battery_type:str):
        """
        :param input_path: the path of data file
        :param file_name: https://batteryarchive.org/metadata.html Naming rules:
        institution
        code_(original cell ID)_form factor_cathode_temperature_min-max SOC_charge rate/discharge rate_(letter)
        :param saving_path: the output path of results
        :param battery_type: battery in different companies
        """
        self.input_path = input_path
        self.saving_path = saving_path
        mkdir(self.saving_path)
        self.file_name = file_name
        norminal_capacity = {'LFP': 1.1, 'NCA': 3.2, 'NMC': 3}
        self.norminal_capacity = norminal_capacity[battery_type]
        self.cycle_data, self.time_series = self.load_data()
        self.font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 32}

    def load_data(self):
        """
        load cycle data and time series
        :return:
        """
        cycle_data = pd.read_csv(filepath_or_buffer=os.path.join(self.input_path, self.file_name+'_cycle_data.csv'))
        cycle_data['Discharge_soh'] = cycle_data['Discharge_Capacity (Ah)'] / self.norminal_capacity
        cycle_data['Cycle_Index'] = cycle_data['Cycle_Index'].apply(lambda x: int(x))
        cycle_data = cycle_data.drop_duplicates()
        time_series = pd.read_csv(filepath_or_buffer=os.path.join(self.input_path, self.file_name+'_timeseries.csv'))
        time_series['Cycle_Index'] = time_series['Cycle_Index'].apply(lambda x: int(x))
        return cycle_data, time_series

    def fully_discharge(self):
        """
        find a capacity check in each period for SNL battery dataset
        https://batteryarchive.org/study_summaries.html
        :return:
        """
        # find the boundary of period
        boundary_flags = self.cycle_data['Cycle_Index'][(self.cycle_data['Discharge_soh'] < 2) &
                                                        (self.cycle_data['Discharge_soh'] > 1.0)]
        boundary_flags.drop(index=[523, 894], inplace=True)
        fully_dchg_flags = self.cycle_data['Cycle_Index'][(self.cycle_data['Discharge_soh'] < 1) &
                                                          (self.cycle_data['Discharge_soh'] > 0.7)]
        fully_dchg = self.cycle_data[self.cycle_data['Discharge_soh'] > 0.7]
        dchg_soh = fully_dchg[['Cycle_Index', 'Discharge_soh']]
        # fully discharge capacity
        fully_dchg_capacity = []
        fully_dchg_index = []
        for boundary_flag in boundary_flags:
            fully_dchg_capacity.append(self.cycle_data['Discharge_Capacity (Ah)'][
                                           self.cycle_data['Cycle_Index'] == (boundary_flag-1)].iloc[0])
            fully_dchg_index.append(boundary_flag-1)

        # last period
        fully_dchg_capacity.append(self.cycle_data['Discharge_Capacity (Ah)'][
                                   self.cycle_data['Cycle_Index'] == fully_dchg_flags.iloc[-2]].iloc[0])
        fully_dchg_index.append(int(fully_dchg_flags.iloc[-2]))

        return fully_dchg_capacity, fully_dchg_index

    def shallow_charge_curve(self, plot_setting=False):
        fully_chg_capacity, fully_chg_index = self.fully_discharge()
        cycle_soh = self.cycle_data[['Cycle_Index', 'Discharge_soh']]
        cycle_soh = cycle_soh.set_index('Cycle_Index')
        fully_chg_curve, shallow_chg_curve = [], []
        for (start_cycle_idx, chg_capacity) in zip(fully_chg_index, fully_chg_capacity):
            # extract fully charge curve
            temp_fully_chg_curve = self.time_series[self.time_series['Cycle_Index'] == start_cycle_idx]
            temp_fully_chg_curve['Calibrate_capacity'] = chg_capacity
            fully_chg_curve.append(temp_fully_chg_curve[temp_fully_chg_curve['Current (A)'] > 1.5])
            # extract shallow charge curve
            for increased_idx in range(20):
                if start_cycle_idx == fully_chg_index[-1]:
                    pending_cycle_idx = start_cycle_idx - increased_idx
                else:
                    pending_cycle_idx = increased_idx + start_cycle_idx
                if ((cycle_soh.loc[pending_cycle_idx][0] > 0.4) & (cycle_soh.loc[pending_cycle_idx][0] < 0.7)):
                    temp_shallow_chg_curve = self.time_series[self.time_series['Cycle_Index']==pending_cycle_idx]
                    temp_shallow_chg_curve['Calibrate_capacity'] = chg_capacity
                    # The current is positive when the battery is charging
                    shallow_chg_curve.append(temp_shallow_chg_curve[temp_shallow_chg_curve['Current (A)'] > 1.5])
                    break
                else:
                    continue

        if plot_setting:
            # plot the voltage curve in shallow charge
            fig = plt.figure(figsize=(16, 12))
            ax = fig.add_subplot(111)
            for curve in shallow_chg_curve:
                ax.plot(curve['Voltage (V)'].values)
            # ax.set_xlabel('Time', fontdict=self.font)
            ax.set_ylabel('Voltage (V)', fontdict=self.font)
            ax.set_title('Voltage', fontdict=self.font)
            plt.show()

            # plot the capacity degradation curve in shallow charge
            fig = plt.figure(figsize=(16, 12))
            ax = fig.add_subplot(111)
            for curve in shallow_chg_curve:
                ax.plot(fully_chg_capacity)
            # ax.set_xlabel('Time', fontdict=self.font)
            ax.set_ylabel('Capacity (Ah)', fontdict=self.font)
            ax.set_title('Capacity degradation', fontdict=self.font)
            plt.show()

            # plot the voltage vs capacity curve in shallow charge
            fig = plt.figure(figsize=(16, 12))
            ax = fig.add_subplot(111)
            for curve in shallow_chg_curve:
                ax.plot(curve['Voltage (V)'].values, curve['Charge_Capacity (Ah)'].values)
            ax.set_xlabel('Voltage (V)', fontdict=self.font)
            ax.set_ylabel('Capacity (Ah)', fontdict=self.font)
            ax.set_title('Voltage vs. Capacity', fontdict=self.font)
            plt.show()

        # save shallow and fully charge curve
        charge_curve = [shallow_chg_curve, fully_chg_curve]
        with open(self.saving_path+self.file_name+'_charge_curve.pk', 'wb') as f:
            pickle.dump(charge_curve, f)

    def fully_charge_curve(self, plot_setting=False):
        # boundary_flags = self.cycle_data['Cycle_Index'][(self.cycle_data['Discharge_soh'] < 0.83) &
        #                                                 (self.cycle_data['Discharge_soh'] > 0.75)]
        fully_dchg_flags = self.cycle_data['Cycle_Index'][(self.cycle_data['Discharge_soh'] < 1) & (self.cycle_data['Discharge_soh'] > 0.7)]
        data = self.cycle_data

        # fully discharge capacity
        fully_chg_capacity = [self.cycle_data['Discharge_Capacity (Ah)'][
                                           self.cycle_data['Cycle_Index'] == dchg_flag].iloc[0]
                              for dchg_flag in fully_dchg_flags]
        fully_chg_curve= []
        for (start_cycle_idx, chg_capacity) in zip(fully_dchg_flags, fully_chg_capacity):
            # extract fully charge curve
            temp_fully_chg_curve = self.time_series[self.time_series['Cycle_Index'] == start_cycle_idx]
            temp_fully_chg_curve['Calibrate_capacity'] = chg_capacity
            fully_chg_curve.append(temp_fully_chg_curve[temp_fully_chg_curve['Current (A)'] > 1.55])

        if plot_setting:
            # plot the voltage curve in shallow charge
            fig = plt.figure(figsize=(16, 12))
            ax = fig.add_subplot(111)
            for curve in fully_chg_curve:
                ax.plot(curve['Voltage (V)'].values)
            # ax.set_xlabel('Time', fontdict=self.font)
            ax.set_ylabel('Voltage (V)', fontdict=self.font)
            ax.set_title('Voltage', fontdict=self.font)
            plt.show()

            # plot the capacity degradation curve in shallow charge
            fig = plt.figure(figsize=(16, 12))
            ax = fig.add_subplot(111)
            for curve in fully_chg_curve:
                ax.plot(fully_chg_capacity)
            # ax.set_xlabel('Time', fontdict=self.font)
            ax.set_ylabel('Capacity (Ah)', fontdict=self.font)
            ax.set_title('Capacity degradation', fontdict=self.font)
            plt.show()

            # plot the voltage vs capacity curve in shallow charge
            fig = plt.figure(figsize=(16, 12))
            ax = fig.add_subplot(111)
            for curve in fully_chg_curve:
                ax.plot(curve['Voltage (V)'].values, curve['Charge_Capacity (Ah)'].values)
            ax.set_xlabel('Voltage (V)', fontdict=self.font)
            ax.set_ylabel('Capacity (Ah)', fontdict=self.font)
            ax.set_title('Voltage vs. Capacity', fontdict=self.font)
            plt.show()

        fully_chg_curve = [[], fully_chg_curve]
        # save shallow and fully charge curve
        with open(self.saving_path+self.file_name+'_charge_curve.pk', 'wb') as f:
            pickle.dump(fully_chg_curve, f)