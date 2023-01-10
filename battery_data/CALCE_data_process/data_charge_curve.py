# coding = utf-8
"""
作者   : Hilbert
时间   :2022/7/18 21:59
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

input_path = '../battery_data/data_curve'

# for data_file in os.listdir(input_path):
# # for data_file in data_files:
#     if 'charge' in data_file:
#         data_name = data_file.split('_')[0]
#         saving_path = os.path.join(saving_path, data_name)
#         mkdir(saving_path)
#         charge_curve = BatteryDataAnalysis(input_path=input_path, saving_path=saving_path, file_name=data_name)
#         charge_curve.fully_curve_plot()

# data_files = ['PL23', 'PL03', 'PL10', 'PL04', 'PL05', 'PL19', 'PL24', 'PL11', 'PL13']
data_files = ['PL23']
for data_name in tqdm(data_files):
# for data_file in data_files:
    saving_path = './result/raw_data_analysis/'
    saving_path = os.path.join(saving_path, data_name, 'charge_data')
    mkdir(saving_path)
    charge_curve = BatteryDataAnalysis(input_path=input_path, saving_path=saving_path, file_name=data_name)
    # charge_curve.fully_curve_plot()
    charge_curve.shallow_curve_plot()

# data_files = ['PL11', 'PL13']
# for data_name in tqdm(data_files):
# # for data_file in data_files:
#     saving_path = './result/raw_data_analysis/'
#     saving_path = os.path.join(saving_path, data_name, 'charge_data')
#     mkdir(saving_path)
#     charge_curve = BatteryFullyDataAnalysis(input_path=input_path, saving_path=saving_path, file_name=data_name)
#     # charge_curve.fully_curve_plot()
#     charge_curve.fully_charge_plot()