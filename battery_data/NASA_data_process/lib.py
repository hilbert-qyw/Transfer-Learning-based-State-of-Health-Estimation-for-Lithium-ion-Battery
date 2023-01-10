# coding = utf-8
"""
作者   : Hilbert
时间   :2022/3/1 10:41
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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import datetime


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


class BatteryDataAnalysis(object):
    def __init__(self, input_path: str, saving_path: str, file_name: str):
        self.input_path = input_path
        self.saving_path = saving_path
        self.name = file_name
        self.dataset, self.capacity = self.load_data()
        self.font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 32}


    def load_data(self):
        """
        Data Structure:
        cycle:	top level structure array containing the charge, discharge and impedance operations
        type: 	operation  type, can be charge, discharge or impedance
        ambient_temperature:	ambient temperature (degree C)
        time: 	the date and time of the start of the cycle, in MATLAB  date vector format
        data:	data structure containing the measurements
	    for charge the fields are:
            Voltage_measured: 	Battery terminal voltage (Volts)
            Current_measured:	Battery output current (Amps)
            Temperature_measured: 	Battery temperature (degree C)
            Current_charge:		Current measured at charger (Amps)
            Voltage_charge:		Voltage measured at charger (Volts)
            Time:			Time vector for the cycle (secs)
	    for discharge the fields are:
            Voltage_measured: 	Battery terminal voltage (Volts)
            Current_measured:	Battery output current (Amps)
            Temperature_measured: 	Battery temperature (degree C)
            Current_charge:		Current measured at load (Amps)
            Voltage_charge:		Voltage measured at load (Volts)
            Time:			Time vector for the cycle (secs)
            Capacity:		Battery capacity (Ahr) for discharge till 2.7V
	    for impedance the fields are:
            Sense_current:		Current in sense branch (Amps)
            Battery_current:	Current in battery branch (Amps)
            Current_ratio:		Ratio of the above currents
            Battery_impedance:	Battery impedance (Ohms) computed from raw data
            Rectified_impedance:	Calibrated and smoothed battery impedance (Ohms)
            Re:			Estimated electrolyte resistance (Ohms)
            Rct:			Estimated charge transfer resistance (Ohms)
        :return:
        """

        mat = sio.loadmat(self.input_path + self.name + '.mat')
        print('Total data in dataset: ', len(mat[self.name][0, 0]['cycle'][0]))
        counter = 0
        dataset = []
        capacity_data = []

        for i in range(len(mat[self.name][0, 0]['cycle'][0])):
            row = mat[self.name][0, 0]['cycle'][0, i]
            if row['type'][0] == 'discharge':
                ambient_temperature = row['ambient_temperature'][0][0]
                date_time = datetime.datetime(int(row['time'][0][0]),
                                              int(row['time'][0][1]),
                                              int(row['time'][0][2]),
                                              int(row['time'][0][3]),
                                              int(row['time'][0][4])) + datetime.timedelta(seconds=int(row['time'][0][5]))
                data = row['data']
                capacity = data[0][0]['Capacity'][0][0]
                for j in range(len(data[0][0]['Voltage_measured'][0])):
                    voltage_measured = data[0][0]['Voltage_measured'][0][j]
                    current_measured = data[0][0]['Current_measured'][0][j]
                    temperature_measured = data[0][0]['Temperature_measured'][0][j]
                    current_load = data[0][0]['Current_load'][0][j]
                    voltage_load = data[0][0]['Voltage_load'][0][j]
                    time = data[0][0]['Time'][0][j]
                    if j == 0:
                        soc = capacity / capacity
                    else:
                        soc = soc + (current_measured * (time - data[0][0]['Time'][0][j-1]) / 3600) / capacity

                    dataset.append([counter + 1, ambient_temperature, date_time, capacity, soc,
                                    voltage_measured, current_measured,
                                    temperature_measured, current_load,
                                    voltage_load, time])
                capacity_data.append([counter + 1, ambient_temperature, date_time, capacity])
                counter = counter + 1
        print(dataset[0])
        return [pd.DataFrame(data=dataset,
                             columns=['cycle', 'ambient_temperature', 'datetime',
                                      'capacity', 'soc', 'voltage_measured',
                                      'current_measured', 'temperature_measured',
                                      'current_load', 'voltage_load', 'time']),
                pd.DataFrame(data=capacity_data,
                             columns=['cycle', 'ambient_temperature', 'datetime',
                                      'capacity'])]


    def sensor_time(self, sensor_type):
        # voltage and current curve

        # delete value less than a certain value
        cycle_len = []
        index = self.dataset['current_measured'][(self.dataset['current_measured'].abs() < 1.75) |
                                                 (self.dataset['current_measured'].abs() > 2.25)].index.tolist()
        self.dataset.drop(index=index, inplace=True)
        cycle = self.dataset['cycle'].unique()
        fig1 = plt.figure(figsize=(16, 12))
        ax1 = fig1.add_subplot(111)
        ax1.grid(alpha=0.4, linestyle=':')
        for idx in cycle:
            sensor_data = self.dataset.loc[self.dataset['cycle'][self.dataset['cycle'] == idx].index]
            cycle_len.append(len(sensor_data))
            ax1.plot(sensor_data['time'], sensor_data[sensor_type+'_measured'], lw=2)
        ax1.set_xlabel('Time', fontdict=self.font)
        ax1.set_ylabel(sensor_type, fontdict=self.font)
        ax1.set_title(sensor_type + ' changes with time', fontdict=self.font)
        plt.tight_layout()
        fig1.savefig(self.saving_path + sensor_type + ' changes with time.png')
        plt.show()

    def plot_capacity(self):
        fig1 = plt.figure(figsize=(16, 12))
        ax1 = fig1.add_subplot(111)
        ax1.grid(alpha=0.4, linestyle=':')
        ax1.plot(self.capacity['capacity'], lw=2)
        ax1.set_xlabel('Time', fontdict=self.font)
        ax1.set_ylabel('capacity', fontdict=self.font)
        ax1.set_title('Capacity decay', fontdict=self.font)
        plt.tight_layout()
        fig1.savefig('Capacity varies with cycle.png')
        plt.show()