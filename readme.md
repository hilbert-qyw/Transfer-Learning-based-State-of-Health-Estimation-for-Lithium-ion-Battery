# Transfer Learning-based State of Health Estimation for Lithium-ion Battery
Author: Yuwen Qin, Xin Chen

## Background
Due to the continuous improvements in their cycling efficiencies and energy and power densities, as well as the reduction in their costs, lithium-ion (Li-ion) rechargeable batteries have been increasingly used as energy storage devices in various applications such as cell phones, laptops, hybrid and electric vehicles, and grid-scale energy storage systems. In such applications, it is of great importance to be able to online estimate the state of health (SOH) of a battery cell for ensuring its safety and reliability when operating in the field. Accurately estimating the capacity of a cell can potentially assist in detecting a faulty condition in the cell and then enable its failure prevention through timely maintenance actions.

In real battery applications, the frequently changing load and operating conditions, different cell chemistries and formats, and complicated degradation patterns still pose a major challenge to expanding the applications of these methods in the battery management field. Urgent efforts are needed to improve the evolving data-driven methods and make them more efficient and robust for different battery applications. To achieve this, by transferring the existing knowledge from different but related domains to the target of interest, transfer learning (TL) is becoming one of the most promising strategies for smarter and more efficient battery management. Specifically, the main hindered relationships learned by the hidden layers can be retained by freezing these layers, while the new characteristics are quickly learned via retraining the last few fully connected layers, which benefits the accuracy and supports online training for battery states estimation and aging prognostics<sup>[1-5]</sup>.



## Data

All the information about battery data comes from https://www.batteryarchive.org/.

### NASA Battery Dataset

Also find in https://data.nasa.gov/dataset/Li-ion-Battery-Aging-Datasets/uj5r-zjdb.

#### Overview

This data set has been collected from a custom built battery prognostics testbed at the NASA Ames Prognostics Center of Excellence (PCoE). Li-ion batteries were run through 3 different operational profiles (charge, discharge and Electrochemical Impedance Spectroscopy) at different temperatures. Discharges were carried out at different current load levels until the battery voltage fell to preset voltage thresholds. Some of these thresholds were lower than that recommended by the OEM (2.7 V) in order to induce deep discharge aging effects. Repeated charge and discharge cycles result in accelerated aging of the batteries. The experiments were stopped when the batteries reached the end-of-life (EOL) criteria of 30% fade in rated capacity (from 2 Ah to 1.4 Ah).

#### Study Conditions

NASA battery dataset contains different Li-ion batteries, and each Li-ion battery repeats three operations (charge, discharge, and impedance measurements). Take batteries #5 for an example.

Fig. 1 shows the #5 Li-ion charge and discharge process in one cycle. Obviously, the charge process consists of constant current (CC) mode and constant voltage (CV) mode. In the charge in CC mode, the current is kept at 1.5 A until the Li-ion battery voltage is increased to 4.2 V. In the charge CV mode, the voltage holds 4.2 V until the Li-ion battery current drops to 20 mA from 1.5 A. In the whole charge process, the battery terminal voltage, battery output current, battery temperature, measured current, and measured voltage are recorded. The discharge process belongs to the CC mode, and the current is 2 A until the Li-ion battery voltage drops to 2.7 V from 4.2 V. In the discharge process, the recorded variables (except battery capacity) are the same as those of the charge process. Meanwhile, NASA Ames utilizes the electrochemical impedance spectroscopy method (frequency sweep from 0.1 Hz to 5 kHz) to measure impedance. In the impedance process, the sensor current, battery current, and ratio of the above currents are recorded to calculate battery impedance, electrolyte resistance (Re), and charge transfer resistance. As time goes on, the repeated charge and discharge process results in accelerated degradation, and eventually, the end of the battery’s service life.

![image-20230109103904103](C:\Users\11855\AppData\Roaming\Typora\typora-user-images\image-20230109103904103.png)

<center>Fig. 1 . Visualization of Li-ion charge and discharge process in one cycle<center>

### NASA Randomized Battery Usage Dataset

Also find in https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository

#### Overview

The NASA randomized battery usage dataset contains data for 28 lithium cobalt oxide (LCO) 18650 cells with a nominal capacity of ∼2.2 Ah. The cells in this dataset were continuously operated. The dataset consists of 7 groups of 4 cells each group cycled at a set ambient temperature (room temp, 40° C); for 5 of these groups the cells were CC-charged to a fixed voltage and then discharged with currents selected at random from the group’s discharge distribution table (7 different regimes). The other two groups were randomly charged and discharged. The dataset includes in-cycle measurements of terminal current, voltage and cell temperature, and measurements of discharging capacity and EIS impedance readings at 50 cycle intervals. The dataset is provided in a ‘.mat’ format and measurements appear to have been taken until the cells reached between 80% to 50% SOH.

#### Study Conditions

1. Selecting a charging or discharging current at random from the set {-4.5A, -3.75A, -3A, -2.25A, -1.5A, -0.75A, 0.75A, 1.5A, 2.25A, 3A, 3.75A, 4.5A}. Negative currents are associated with charging and positive currents indicate discharging.

2. The selected current setpoint is applied until either the battery voltage goes outside the range (3.2V - 4.2V) or 5 minutes has passed. These steps are identified with the **comment** field = **“discharge (random walk)”** and **comment** field = **“charge (random walk)”**
3. After each charging or discharging period, there will be a <1s period of rest while a new charging or discharging current setpoint is selected.
4. This steps is identified with the **comment** field = **“rest (random walk)”
5. Steps 2 and 3 are repeated 1500 times, then characterization cycles are preformed to benchmark battery state of health.

#### Batteries are cycled using three types of reference profiles:

1. A low current discharge at 0.04A is used to observe the batteries open circuit voltage as a function of SOC
   - This profile is identified in the **step** data structure with the **comment** field equal to **“low current discharge at 0.04A”**
   - Resting periods that occur immediately before and after the low current discharge are identified with the **comment** field = **“rest prior low current discharge”** and **comment** field = **“rest post low current discharge”** respectively
2. A reference charge and discharge cycle is used to observe the battery capacity after every 1500 RW steps cycles
   - Batteries are first charged at 2A (constant current), until they reach 4.2V, at which time the charging switches to a constant voltage mode and continues charging the batteries until the charging current falls below 0.01A.
     - This step is identified with the **comment** field = **“reference charge”**
   - Batteries are then rested for a period of time with no current draw
     - This step is identified with the **comment** field = **“rest post reference charge”**
   - Batteries are then discharged at 2A until the battery voltage crosses 3.2V
     - This step is identified with the **comment** field = **“reference discharge”**
   - Batteries are then rested for a period of time
     - This step is identified with the **comment** field = **“rest post reference discharge”**
3. A pulsed current discharge of fully charged batteries is performed after every 3000 RW steps in order to benchmark changes to battery transient dynamics. The pulsed current discharge consist of a 1A load applied for 10 minutes, followed by 20 minutes of no load.
   - This discharging profile is identified by alternating steps of **comment** = **“pulsed load (rest)”** and **comment** = **“pulsed load (discharge)”**
   - A resting period after either a pulsed discharge or a pulsed charge is denoted by **comment** = **“rest post pulsed load or charge”**
4. A pulsed current charge of recently discharged batteries is performed after every 3000 RW cycles in order to benchmark changes to battery transient dynamics. The pulsed current charge consists of a 1A charging current applied for 10 minutes, followed by 20 minutes of rest.
   - This charging profile is identified by alternating steps of **comment** = **“pulsed charge (rest)”** and **comment** = **“pulsed charge (charge)”**
   - A resting period after either a pulsed discharge or a pulsed charge is denoted by **comment** = **“rest post pulsed load or charge”**



### Center for Advanced Life Cycle Engineering (CALCE)[<sup>5</sup>](#refer-anchor-5)

Also find in https://web.calce.umd.edu/batteries/data.htm.

#### Overview

Lithium-ion batteries are used for energy storage in a wide array of applications, and do not always undergo full charge and discharge cycling. We conducted an experiment which **quantifies the effect of partial charge-discharge cycling on Li-ion battery capacity loss** by means of cycling tests conducted on graphite/LiCoO2 pouch cells under different state of charge (SOC) ranges and discharge currents. **The results are used to develop a model of capacity fade** for batteries under full or partial cycling conditions. This experiment demonstrates that all of the variables studied including mean SOC, change in SOC (ΔSOC) and discharge rate have a significant impact on capacity loss rate during the cycling operation. The initial characterization tests for the cells included constant current constant voltage (CCCV) charge - constant current (CC) full discharge (4.2V - 2.7V) at C/2 rate to determine battery discharge capacity. 

#### Study Conditions

1. Cells were initially charged to 100% SOC using the CCCV profile at C/2 rate.
2. After reaching 100% SOC, the cells were discharged using constant C/2 current until they reached the lower limits of their assigned SOC ranges (i.e., 20% for 20% - 80% range) for partial cycling.
3. Constant current charge (always C/2) and constant current discharge (C/2 or 2C) were applied to the cells for cycling between the desired upper and lower limits of SOC (i.e., 20% - 80%).
4. A rest period of 30 min was applied to allow the cells to relax after every charge and discharge steps.



### Sandia National Laboratories Study Overview(SNL)[<sup>8</sup>](#refer-anchor-8)

#### Overview

The dataset from Sandia National Labs used in the publication, “Degradation of Commercial Lithium-ion Cells as a Function of Chemistry and Cycling Conditions,” consists of commercial 18650 NCA, NMC, and LFP cells cycled to 80% capacity (although cycling is still ongoing). This study examines the influence of temperature, depth of discharge (DOD), and discharge current on the long-term degradation of the commercial cells.

#### Study Conditions

Cycle aging was carried out using an Arbin SCTS and an Arbin high-precision (Model: LBT21084) multi-channel battery testing system. Individual cells were placed into commercially available 18650 battery holders (Memory Protection Devices). The holders were connected to the Arbin with 18 gauge wire and the cable lengths kept below eight feet to minimize voltage drop. During cycling, the cells were placed in SPX Tenney Model T10C-1.5 environmental chambers. A K-type thermocouple was attached to the skin of each cell under test with Kapton tape to monitor the cell skin temperature.



## Requirements

```
numpy~=1.19.2
pandas~=1.4.0
matplotlib~=3.3.2
torch~=1.9.0
scikit-learn~=0.23.2
tqdm~=4.50.2
scipy~=1.5.2
xlrd~=1.2.0
openpyxl~=3.0.5
```

Dependencies can be installed using the following command:

```python
pip install -r requirements.txt
```



## Fine-tuning strategy

Fig. 2 illustrates a typical process of fine-tuning strategy-based battery SOH estimation[<sup>6</sup>](#refer-anchor-6). The non-linear mapping between input data such as current, voltage, temperature, and output SOH is learned by training a machine learning model. There are two ways to fine-tune the parameters of a data-driven model for battery SOH estimation. The first is to treat the parameters of a pre-trained model as initial values for the target battery. All the parameters from a pre-trained data-driven model can serve as prior knowledge when new data from the target domain are used to re-train this model. With this prior knowledge, the retraining process can be accelerated as this knowledge makes it easier to find the new local optimum when the data-driven model is re-trained for the target task. Another way to realize fine-tuning strategy-based TL for battery SOH estimation is to fix some parameters of the pre-trained model and fine-tuning other parameters for new applications. For neural network cases, either shallow layers or top layers of the neural network could be frozen and the other in the network can be set as adjustable. 

![image-20230110105846520](C:\Users\11855\AppData\Roaming\Typora\typora-user-images\image-20230110105846520.png)

<center>Fig. 1 . A typical process of fine-tuning strategy<center>

​    

## Project Organization

```
├── battery_data
│   ├── NASA_data_process
│   │   ├── data_visualization.py 		<- Process NASA battery dataset and visualization
│   │   ├── lib.py						<- Module contains loading data and analysing data for NASA dataset
│
│   ├── CALCE_data_process
│   │   ├── data_transformation.py		<- Transform CALCE battery dataset from .xlsx format to .pk format
│   │   ├── data_segment.py				<- Extract charging curve from all cycles and save file
│   │   ├── data_charge_curve.py        <- Visualize fully charging curve and shallow charging curve
│   │   ├── lib.py						<- Module contains data segmentation and data analysis for CALCE 											   dataset
│
│   ├── SNL_data_process
│   │   ├── data_segmentation.py		<- Extract charging curve from all cycles and save file
│   │   ├── cycle_data.py				<- Visualize the degradation profile of SNL battery dataset 
│   │   ├── data_lib.py					<- Module contains data segmentation for SNL dataset
│
│
├── model
│   ├── battery_capacity_training.py	<- Train battery capacity estimation model
│   ├── battery_capacity_test.py		<- Restore model and test performance
│   ├── battery_transfer.py				<- Transfer learning with fine-tuning strategy
│   ├── transfer_comparative_experiment.py	<- Discuss the effect of different transfer learning parameter 												   settings
│   ├── transfer_random_search.py		<- Random search selects the optimal hyperparameter configuration
│   ├── condition_cluster.py			<- t-SNE cluster visualizes the difference of battery degradation 											   features in various experimental conditions
│   ├── model_lib.py					<- Module contains various NN models
│   ├── informer.py						<- Module contains the structure of Informer NN model
│   ├──	utils.py						<- Module contains network dataset, dataloader, trainer, and tester 
```



## Reference

<div id="refer-anchor-1"></div>

- [1] Liu, Kailong, et al. "Transfer learning for battery smarter state estimation and ageing prognostics: Recent progress, challenges, and prospects." *Advances in Applied Energy* (2022): 100117.

<div id="refer-anchor-2"></div>

- [2] Wang, Cunsong, et al. "Dynamic long short-term memory neural-network-based indirect remaining-useful-life prognosis for satellite lithium-ion battery." *Applied Sciences* 8.11 (2018): 2078.

<div id="refer-anchor-3"></div>

- [3] Fan, Cheng, et al. "Statistical investigations of transfer learning-based methodology for short-term building energy predictions." *Applied Energy* 262 (2020): 114499.

<div id="refer-anchor-4"></div>

- [4] Ma, Guijun, et al. "A Transfer Learning-Based Method for Personalized State of Health Estimation of Lithium-Ion Batteries." *IEEE Transactions on Neural Networks and Learning Systems* (2022).

  <div id="refer-anchor-5"></div>

- [5] Zhou, Kate Qi, Yan Qin, and Chau Yuen. "Transfer-Learning-Based State-of-Health Estimation for Lithium-Ion Battery With Cycle Synchronization." *IEEE/ASME Transactions on Mechatronics* (2022).

<div id="refer-anchor-6"></div>

- [6] Shen, Sheng, et al. "Deep convolutional neural networks with ensemble learning and transfer learning for capacity estimation of lithium-ion batteries." *Applied Energy* 260 (2020): 114296.
