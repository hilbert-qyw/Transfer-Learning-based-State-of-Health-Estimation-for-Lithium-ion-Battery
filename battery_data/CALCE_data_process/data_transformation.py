# coding = utf-8
"""
作者   : Hilbert
时间   :2022/7/13 22:04
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


input_path = fr"../battery_data/data_xlsx"
saving_path = '../battery_data/data_pickle'
mkdir(saving_path)
# data_name = 'PL18'

# for file_name in os.listdir(input_path):
#     data_name = file_name.split('.xlsx')[0]
#     print(fr"Processing {data_name}...")
#     mat2pk(input_path, saving_path, data_name)

file_names = ['PL11.xlsx', 'PL13.xlsx']
for file_name in file_names:
    data_name = file_name.split('.xlsx')[0]
    print(fr"Processing {data_name}...")
    mat2pk(input_path, saving_path, data_name)