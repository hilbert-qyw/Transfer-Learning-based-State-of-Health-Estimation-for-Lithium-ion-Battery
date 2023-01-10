# coding = gbk
"""
作者   : Hilbert
时间   :2022/3/1 10:54
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


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from lib import *


input_path = r"E:\hilbert_研究生\项目\华电电池储能\battery_dataset\NASA_battery_dataset\BatteryAgingARC-05_06_07_18\\"
saving_path = './result/'
mkdir(saving_path)
data_name = 'B0007'

battery_data = BatteryDataAnalysis(input_path=input_path, saving_path=saving_path, file_name=data_name)
battery_data.plot_capacity()
battery_data.sensor_time(sensor_type='voltage')
battery_data.sensor_time(sensor_type='current')
battery_data.sensor_time(sensor_type='temperature')