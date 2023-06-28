Regarding the research on fall prediction and fall detection, you can refer to our [comprehensive review](https://doi.org/10.1016/j.engappai.2023.105993).
This work can be cited as BibTex:
```bib
@article{LIU2023105993,
title = {A review of wearable sensors based fall-related recognition systems},
journal = {Engineering Applications of Artificial Intelligence},
volume = {121},
pages = {105993},
year = {2023},
issn = {0952-1976},
doi = {https://doi.org/10.1016/j.engappai.2023.105993},
url = {https://www.sciencedirect.com/science/article/pii/S095219762300177X},
author = {Jiawei Liu and Xiaohu Li and Shanshan Huang and Rui Chao and Zhidong Cao and Shu Wang and Aiguo Wang and Li Liu}
}
```


# MCTN
Repo of Paper " MCTN: A MULTI-CHANNEL TEMPORAL NETWORK FOR WEARABLE FALL PREDICTION" in ECML PKDD 2023

This repository contains codes for a Transformer-based model that employs Positional Embedding (PE) and Channel Embedding (CE) to capture the spatio-temporal characteristics between multiple sensor data. Our objective is to predict possible falls from these human activity data with spatio-temporal characteristics. The overview of our approach is illustrated in the main paper.  The methods are described in this paper.

## 1. Installation

Our codes are divided into five parts:

1. Tools  (**utils.py** and **features.py**), including loading raw data, feature extraction, PE and CE.
2. Raw data to PTH file in rawData2pth directory.
3. Dataset loading (**sisfall_dataset.py**, **mobiact_dataset.py**,  and **softfall_dataset.py**), implemented separately for each of the three datasets.
4. MCTN model (**mctn_model.py**) which is implemented by PyTorch. 
5. Parameters of the experiment (**paras.py**), including lead time, window time, sampling rate, MCTN parameters, etc.

To run the code, please clone this repository. You need to install the environments: `Python 3.8+` including the packages  `torch>=1.8.1`, and`numpy>=1.23.3`.

## 2. Dataset Format

Experiments are carried out on two publicly-available benchmarks and one in-house dataset `SoftFall` collected by ourselves.

**Raw data**

1. **SisFall (200Hz)**. This publicly-available dataset contains a total of 4505 samples, including 2707 samples of 19 activities of daily life (ADL) and 1798 samples of 15 fall actions collected from two accelerometers and one gyroscope placed at the waist of 38 participants. 
2. **MobiAct (100Hz)**. It is a public benchmark that contains 647 samples of 4 fall actions and 1879 samples of 9 ADL actions obtained by an accelerometer and a gyroscope placed in the thighs of 57 participants.
3. **SoftFall (25Hz)**. To the best of our knowledge, these two publicly-available datasets do not contain the scenario of falls of the elderly where their procedure is often slower than and different from the other cases. To this end, we propose a new dataset that includes simulated slow-paced movements of elderly individuals (see **Table 1**). These samples were collected using a 9-axis MPU9250 sensor (one accelerometer, one gyroscope and one magnetometer) attached at the subject’s waist from 11 participants, who were instructed to perform 22 different activities (7 ADL activities and 15 fall activities). A total number of 802 samples were collected, including 203 ADL samples and 599 fall samples.

**Table 1**. Fall and ADL events in SoftFall. All ADLs are slow-paced. Except for the *pre-fall* phase in the falling data, the rest are only affected by gravity without external interference.

<table> 	
<tr> <th align="center">Fall / ADL</th> 	    <th align="center">Number</th> 	    <th align="center">Activity</th> </tr > 	
<tr > <td rowspan="15" align="center">Fall</td> 	    <td align="center">0</td> 	    <td>Basic falls (fall in four directions from front to back, left to right while standing).</td> </tr> 	
<tr><td align="center">1</td> 	    <td>Crouch-uplateral fall. </td> </tr> 	
<tr><td align="center">2</td> 	    <td>Crouch-upbackward fall.</td> 	</tr> 	
<tr><td align="center">3</td> 	    <td>Bending knee to stand up and fall forward.</td> 	</tr> 
<tr><td align="center">4</td> 	    <td>Bending knee to stand up and fall backward.</td> 	</tr> 
<tr><td align="center">5</td> 	    <td>Bending forward to fall.</td> 	</tr>
<tr><td align="center">6</td> 	    <td>Bending sideways and falling.</td> 	</tr> 
<tr><td align="center">7</td> 	    <td>Bending up and falling backward.</td> 	</tr> 
<tr><td align="center">8</td> 	<td>Lying down and roll over to fall vertically. </td> 	</tr> 
<tr><td align="center">9</td> 	    <td>Slipping backward while walking.</td></tr> 
<tr><td align="center">10</td> 	    <td >Tripping in forward direction while walking. </td> </tr> 	
<tr> <td align="center">11</td> 	    <td >Lateral collision while walking.</td></tr> 
<tr> <td align="center">12</td> 	    <td >Fainting directly to the side while walking. </td> </tr> 
<tr> <td align="center">13</td> 	    <td >Tripping in forward direction while running.</td> </tr> 
<tr> <td align="center">14</td> 	    <td >Slipping sideways while running. </td> </tr> 
<tr > <td rowspan="7" align="center">ADL</td> 	    <td align="center">0</td> 	    <td>(Slowly) Flow ADL (a complete set of movements from ADL1 to ADL6).</td> </tr> 	
<tr><td align="center">1</td> 	    <td> (Slowly) Walking.</td> </tr> 	
<tr><td align="center">2</td> 	    <td>(Slowly) Running.</td> 	</tr> 	
<tr><td align="center">3</td> 	    <td>(Slowly) Picking up/bending down.</td> 	</tr> 
<tr><td align="center">4</td> 	    <td>(Slowly) Going up and down stairs.</td> 	</tr> 
<tr><td align="center">5</td> 	    <td>(Slowly) Lying-Sitting-Standing.</td> 	</tr>
<tr><td align="center">6</td> 	    <td>(Slowly) Standing-Sitting-Lying.</td> 	</tr> 
</table>


## 3. Guidelines for Usage

1. To train the MCTN model, run "**mctn_model.py**"  in PyCharm.

**Other Information**

1. Due to the number of trials, we saved the raw dataset as pytorch files, which can be seen in **rawData2pth**.

2. Modify the hyperparameters of the data window and MCTN: Adjust the corresponding hyperparameters in "**paras.py**". The corresponding parameters for selecting the fall prediction window, which is approximately 3 seconds before and 0.6 seconds ahead of time, for different datasets (200Hz and 100Hz) are as follows.

   |  Hz  | sisfall-down/mobiact-down | pre-len | front-len | n-head | d-model |
   | :--: | :-----------------------: | :-----: | :-------: | :----: | :-----: |
   |  20  |           10/5            |   12    |    61     |   7    |   63    |
   |  25  |            8/4            |   15    |    75     |   7    |   77    |
   |  50  |            4/2            |   30    |    143    |   5    |   145   |
   | 100  |            2/1            |   60    |    295    |   9    |   297   |
   | 200  |          1/None           |   120   |    581    |   11   |   583   |

3. After the preparation of the dataset, switching to a different dataset can be achieved by commenting out the unnecessary parts in  "**mctn_model.py**". The loading code for the three datasets used in this study are denoted as a, b, and c, respectively.

   ```python
   my_dataset = MySoftfallData()
   # my_dataset = MyMobiactData()
   # my_dataset = MySisfallData()
   ```

4. By commenting out the unnecessary sections in "**mctn_model.py**", it is possible to utilize data from only two (6-axis) or three (9-axis) sensors.

   ```python
   for step, (train_signal, train_label) in enumerate(train_loader):
       train_signal = train_signal.to(device) # a,g,m / a,g,a
   ```

   ```python
   for (test_signal, test_label) in test_loader:
       # test_signal = test_signal[:, 0:6, ...].to(device) # a,g
       test_signal = test_signal.to(device) # a,g,m / a,g,a
   ```
   
