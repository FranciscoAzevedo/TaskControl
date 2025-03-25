# %% imports 
%matplotlib qt5
%load_ext autoreload
%autoreload 2

# Misc
import os
from pathlib import Path
from tqdm import tqdm

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt

# Math
import scipy as sp
import numpy as np
import pandas as pd
import cv2

# Custom
from Utils import behavior_analysis_utils as bhv
from Utils import dlc_analysis_utils as dlc
import Utils.metrics as metrics
from Utils.sync import Syncer
from Utils import utils

colors = dict(success="#72E043", 
              reward="#3CE1FA", 
              correct="#72E043", 
              incorrect="#F56057", 
              premature="#9D5DF0", 
              missed="#F7D379",
              left=mpl.cm.PiYG(0.05),
              right=mpl.cm.PiYG(0.95))

# %% read all three data sources
session_folder = utils.get_folder_dialog(initial_dir="/media/storage/shared-paton/georg/therapist_testing_sessions_to_keep")

os.chdir(session_folder)

### Camera data
video_path = session_folder / "bonsai_video.avi"
Vid = dlc.read_video(str(video_path))

### Arduino data
log_path = session_folder / 'arduino_log.txt'
LogDf = bhv.get_LogDf_from_path(log_path)

### LoadCell Data
LoadCellDf = bhv.parse_bonsai_LoadCellData_touch(session_folder / 'bonsai_LoadCellData.csv')

# Syncer
from Utils import sync
cam_sync_event, Cam_SyncDf = sync.parse_cam_sync(session_folder / 'bonsai_frame_stamps.csv', offset=1, return_full=True)
lc_sync_event = sync.parse_harp_sync(session_folder / 'bonsai_harp_sync.csv',trig_len=100, ttol=20)
arduino_sync_event = sync.get_arduino_sync(session_folder / 'arduino_log.txt')

Sync = Syncer()
Sync.data['arduino'] = arduino_sync_event['t'].values
Sync.data['loadcell'] = lc_sync_event['t'].values
Sync.data['cam'] = cam_sync_event['t'].values
Sync.data['frames'] = cam_sync_event.index.values

Sync.sync('arduino','cam')
Sync.sync('arduino','loadcell')
Sync.sync('frames','cam')

Sync.eval_plot()

# %%
# preprocessing
# careful: in the bonsai script right now, the baseline removal is different
# should be fixed now

samples = 10000 # 10s buffer: harp samples at 1khz, arduino at 100hz, LC controller has 1000 samples in buffer
for col in tqdm(LoadCellDf.columns):
    if col is not 't':
        LoadCellDf[col] = LoadCellDf[col] - LoadCellDf[col].rolling(samples).mean()

# %%
ds = 10
fig, axes = plt.subplots(nrows=2,sharex=True)
tvec = Sync.convert(LoadCellDf['t'].values, 'loadcell','arduino') / 1e3
axes[0].plot(tvec[::ds],np.abs(LoadCellDf['touch_l'].values[::ds]),color=colors['left'])
axes[0].plot(tvec[::ds],np.abs(LoadCellDf['touch_r'].values[::ds]),color=colors['right'])
axes[0].axhline(100,linestyle=':',color='k')

# tvec = Sync.convert(LoadCellDf['t'].values, 'loadcell','arduino') / 1e3
axes[1].plot(tvec[::ds],LoadCellDf['paw_l'].values[::ds],color=colors['left'])
axes[1].plot(tvec[::ds],LoadCellDf['paw_r'].values[::ds],color=colors['right'])



# %% Plot traces for reach detection
ds = 1
fig, axes = plt.subplots(nrows=3,sharex=True)
tvec = Sync.convert(LoadCellDf['t'].values, 'loadcell','arduino') / 1e3
axes[0].plot(tvec[::ds],np.abs(LoadCellDf['touch_r'].values[::ds]),color=colors['left'])
axes[0].plot(tvec[::ds],np.abs(LoadCellDf['touch_l'].values[::ds]),color=colors['right'])
axes[0].axhline(100,linestyle=':',color='k')

# For each value of a threshold and time see how many buzzer lead to reaches