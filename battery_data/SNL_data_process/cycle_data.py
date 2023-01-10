# -*- coding: gbk -*-
# @Time     : 2022/9/12 22:00
# @Author   : Hilbert
# @Software : PyCharm


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# battery information
cathode = 'LFP'
max_soc = 80
min_soc = 20
discharge_rate = 0.5
charge_rate = 0.5
temperature = 25
battery_idx = 'b'

file_name = fr"SNL_18650_{cathode}_{temperature}C_{min_soc}-{max_soc}_" \
            fr"{charge_rate}-{discharge_rate}C_{battery_idx}_cycle_data.csv"
file_path = r"E:\hilbert_研究生\项目\华电电池储能\程序\Sandia\Battery_archive\battery_dataset"

cycle_data = pd.read_csv(filepath_or_buffer=os.path.join(file_path, file_name))
cycle_data['Discharge_soh'] = cycle_data['Discharge_Capacity (Ah)'] / 1.1
cycle_data['Charge_soh'] = cycle_data['Charge_Capacity (Ah)'] / 1.1
print(cycle_data.shape)

# plot battery capacity degradation profile
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111)
ax.plot(cycle_data['Charge_Capacity (Ah)'], label="charge_curve")
ax.plot(cycle_data['Discharge_Capacity (Ah)'], label="discharge_curve")
plt.legend()
plt.show()

# scatter plot
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111)
ax.scatter(cycle_data['Cycle_Index'], cycle_data['Charge_Capacity (Ah)'], label="charge_curve")
ax.scatter(cycle_data['Cycle_Index'],cycle_data['Discharge_Capacity (Ah)'], label="discharge_curve")
plt.legend()
plt.show()

# fully charge and discharge data
fully_chg = cycle_data[cycle_data['Charge_Capacity (Ah)'] > 0.8]
fully_dchg = cycle_data[cycle_data['Discharge_Capacity (Ah)'] > 0.8]
chg_soh = fully_chg[['Cycle_Index', 'Charge_soh']]
dchg_soh = fully_dchg[['Cycle_Index', 'Discharge_soh']]
print(1)