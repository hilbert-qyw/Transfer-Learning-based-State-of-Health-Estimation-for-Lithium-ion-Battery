# -*- coding: gbk -*-
# @Time     : 2022/9/12 22:38
# @Author   : Hilbert
# @Software : PyCharm

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from data_lib import *

# battery information
cathode = 'NCA'
max_soc = 80
min_soc = 20
discharge_rate = 0.5
charge_rate = 0.5
temperature = 25
battery_idx = 'd'

file_name = fr"SNL_18650_{cathode}_{temperature}C_{min_soc}-{max_soc}_" \
            fr"{charge_rate}-{discharge_rate}C_{battery_idx}"
file_path = r"E:\hilbert_研究生\项目\华电电池储能\程序\Sandia\Battery_archive\battery_dataset"

SNL_dataset = BatteryDataSegment(input_path=file_path, file_name=file_name, saving_path='./results/',
                                 battery_type=cathode)
# fully_dchg_capacity, fully_dchg_index = SNL_dataset.fully_discharge()
SNL_dataset.shallow_charge_curve(plot_setting=True)
# SNL_dataset.fully_charge_curve(plot_setting=True)