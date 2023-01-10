# coding = utf-8
"""
作者   : Hilbert
时间   :2022/7/12 17:13
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


input_path = fr"../battery_data/data_pickle"

# 20-80  0.5C PL21, PL23
# 0-60   0.5C PL03, PL10

# data_files = ['PL21', 'PL23']
# data_files = ['PL03', 'PL10']
# data_files = ['PL04', 'PL05']
# data_files = ['PL19', 'PL24']
# data_files = ['PL11', 'PL13']

# data_files = ['PL21', 'PL23', 'PL03', 'PL10', 'PL04', 'PL05', 'PL19', 'PL24', 'PL11', 'PL13']

data_range = {
    'PL21': {'partial_shallow_chg': [5, 7], 'partial_shallow_dchg': [6, 10], 'partial_fully_chg': [15, 18],
             'partial_fully_dchg': [12, 16], 'single_chg': [8, 11], 'single_dchg': [4, 9],
             'discharge': [1, 5], 'charge': [1, 4]},
    'PL23': {'partial_shallow_chg': [5, 7], 'partial_shallow_dchg': [6, 10], 'partial_fully_chg': [15, 18],
             'partial_fully_dchg': [12, 16], 'single_chg': [8, 11], 'single_dchg': [4, 9],
             'discharge': [1, 5], 'charge': [1, 4]},
    'PL03': {'partial_shallow_chg': [6, 8], 'partial_shallow_dchg': [4, 7], 'partial_fully_chg': [14, 17],
             'partial_fully_dchg': [11, 15], 'single_chg': [8, 11], 'single_dchg': [4, 9],
             'discharge': [1, 5], 'charge': [1, 4]},
    'PL10': {'partial_shallow_chg': [6, 8], 'partial_shallow_dchg': [4, 7], 'partial_fully_chg': [14, 17],
             'partial_fully_dchg': [11, 15], 'single_chg': [8, 11], 'single_dchg': [4, 9],
             'discharge': [1, 5], 'charge': [1, 4]},
    'PL04': {'partial_shallow_chg': [6, 8], 'partial_shallow_dchg': [7, 11], 'partial_fully_chg': [16, 19],
             'partial_fully_dchg': [13, 17], 'single_chg': [8, 11], 'single_dchg': [4, 9],
             'discharge': [1, 5], 'charge': [1, 4]},
    'PL05': {'partial_shallow_chg': [6, 8], 'partial_shallow_dchg': [7, 11], 'partial_fully_chg': [16, 19],
             'partial_fully_dchg': [13, 17], 'single_chg': [8, 11], 'single_dchg': [4, 9],
             'discharge': [1, 5], 'charge': [1, 4]},
    'PL19': {'partial_shallow_chg': [1, 4], 'partial_shallow_dchg': [3, 7], 'partial_fully_chg': [12, 15],
             'partial_fully_dchg': [9, 13], 'single_chg': [8, 11], 'single_dchg': [4, 9],
             'discharge': [1, 5], 'charge': [1, 4]},
    'PL24': {'partial_shallow_chg': [1, 4], 'partial_shallow_dchg': [3, 7], 'partial_fully_chg': [12, 15],
             'partial_fully_dchg': [9, 13], 'single_chg': [8, 11], 'single_dchg': [4, 9],
             'discharge': [1, 5], 'charge': [1, 4]},
    'PL11': {'partial_shallow_chg': [0, 0], 'partial_shallow_dchg': [0, 0], 'partial_fully_chg': [1, 4],
             'partial_fully_dchg': [4, 9], 'single_chg': [0, 0], 'single_dchg': [0, 0],
             'discharge': [1, 5], 'charge': [1, 4]},
    'PL13': {'partial_shallow_chg': [0, 0], 'partial_shallow_dchg': [0, 0], 'partial_fully_chg': [1, 4],
             'partial_fully_dchg': [4, 9], 'single_chg': [0, 0], 'single_dchg': [0, 0],
             'discharge': [1, 5], 'charge': [1, 4]},
}

data_files = ['PL21']
for file in data_files:
    data_name = file.split('.pk')[0]
    saving_path = './result/raw_data_analysis/'
    saving_path = os.path.join(saving_path, data_name)
    mkdir(saving_path)
    battery_data = BatteryDataSegment(input_path=input_path, saving_path=saving_path,
                                       file_name=data_name, data_range=data_range[data_name])
    # battery_data.period_cycle()
    battery_data.each_cycle()