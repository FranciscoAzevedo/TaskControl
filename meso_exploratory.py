# %% imports 
%matplotlib qt5
%matplotlib qt5
%load_ext autoreload
%autoreload 2

# plotting
import matplotlib.pyplot as plt
from matplotlib import cm

# Math
import scipy as sp
import scipy.signal
import numpy as np
import pandas as pd

# Misc
import os
from tqdm import tqdm
from pathlib import Path

# Custom
from Utils import behavior_analysis_utils as bhv
from Utils import dlc_analysis_utils as dlc_utils
from Utils import metrics as met
from Utils import utils
import behav_plotters_reach as bhv_plt_reach

# Settings
# Plotting Defaults
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.major.size"] = 1.5
plt.rcParams["ytick.major.size"] = 1.5
plt.rcParams['figure.dpi'] = 166


# %%
 
"""
 #       #######    #    ######  ### #     #  #####
 #       #     #   # #   #     #  #  ##    # #     #
 #       #     #  #   #  #     #  #  # #   # #
 #       #     # #     # #     #  #  #  #  # #  ####
 #       #     # ####### #     #  #  #   # # #     #
 #       #     # #     # #     #  #  #    ## #     #
 ####### ####### #     # ######  ### #     #  #####

"""

fd_path = utils.get_folder_dialog(initial_dir="/media/storage/shared-paton/georg/mesoscope_testings/behavior")


# Video data
video_path = fd_path / "bonsai_video.avi"
Vid = dlc_utils.read_video(str(video_path))

# Arduino data
log_path = fd_path / 'arduino_log.txt'
LogDf = bhv.get_LogDf_from_path(log_path)

# LoadCell data
LoadCellDf = bhv.parse_bonsai_LoadCellData(fd_path / 'bonsai_LoadCellData.csv')

# Moving average mean subtraction
samples = 1000 # ms
LoadCellDf['x'] = LoadCellDf['x'] - LoadCellDf['x'].rolling(samples).mean()
LoadCellDf['y'] = LoadCellDf['y'] - LoadCellDf['y'].rolling(samples).mean()

## Synching 
from Utils import sync

# Parse sync events/triggers
cam_sync_event = sync.parse_cam_sync(fd_path / 'bonsai_frame_stamps.csv')
lc_sync_event = sync.parse_harp_sync(fd_path / 'bonsai_harp_sync.csv')
arduino_sync_event = sync.get_arduino_sync(fd_path / 'arduino_log.txt')
meso_sync_event = sync.parse_meso_sync(fd_path / 'arduino_log.txt')

# Get the values out of them
Sync = sync.Syncer()
Sync.data['arduino'] = arduino_sync_event['t'].values
Sync.data['loadcell'] = lc_sync_event['t'].values
#Sync.data['dlc'] = cam_sync_event.index.values # the frames are the DLC
#Sync.data['cam'] = cam_sync_event['t'].values # used for what?

# Sync them all to one clock
#Sync.sync('arduino','loadcell')

# Add single GO_CUE_EVENT
LogDf = bhv.add_go_cue_LogDf(LogDf)

#  Create SessionDf - For LEARN_TO_CHOOSE onwards
TrialSpans = bhv.get_spans_from_names(LogDf, "TRIAL_AVAILABLE_STATE", "ITI_STATE")

TrialDfs = []
for i, row in tqdm(TrialSpans.iterrows(),position=0, leave=True):
    TrialDfs.append(bhv.time_slice(LogDf, row['t_on'], row['t_off']))

metrics = (met.get_start, met.get_stop, met.get_correct_side, met.get_interval_category, met.get_outcome, 
            met.get_chosen_side, met.has_reach_left, met.has_reach_right, met.get_in_corr_loop, met.reach_rt_left, 
            met.reach_rt_right, met.has_choice, met.get_interval, met.get_timing_trial, met.get_choice_rt,
            met.get_reached_side, met.get_bias, met.is_anticipatory, met.get_init_rt) 

SessionDf = bhv.parse_trials(TrialDfs, metrics)

# Add choice grasp dur metric computed differently from the other metrics
SessionDf = bhv_plt_reach.compute_choice_grasp_dur(LogDf,SessionDf)

outcomes = SessionDf['outcome'].unique()
for outcome in outcomes:
   SessionDf['is_'+outcome] = SessionDf['outcome'] == outcome


#  Plots dir and animal info
animal_meta = pd.read_csv(log_path.parent.parent / 'animal_meta.csv')
nickname = animal_meta[animal_meta['name'] == 'Nickname']['value'].values[0]
session_date = log_path.parent.stem.split('_')[0]

plot_dir = log_path.parent / 'plots'
os.makedirs(plot_dir, exist_ok=True)

# %%
"""
.########..##........#######..########..######.
.##.....##.##.......##.....##....##....##....##
.##.....##.##.......##.....##....##....##......
.########..##.......##.....##....##.....######.
.##........##.......##.....##....##..........##
.##........##.......##.....##....##....##....##
.##........########..#######.....##.....######.
"""

# %%
# Load meso data already processed from caiman
fd_path = utils.get_folder_dialog(initial_dir="/media/storage/shared-paton/georg/mesoscope_testings")

N = np.load(fd_path / "dff_cnm2_c.npy")
N_coords = np.load(fd_path / "coords.npy") # XY coords of each neuron 

meso_sync_event = meso_sync_event.reset_index(drop=True)

# %% General functions to slice and plot aligned to events

events = ['GO_CUE_EVENT','REWARD_STATE','REWARD_RIGHT_COLLECTED_EVENT']

fig, axes = plt.subplots(ncols=len(events), sharey=True)

pre, post = 500,1000

for i,event in enumerate(events):
    t_refs = bhv.get_events_from_name(LogDf, event).values

    N_mean,frame_time = [],[]
    for t_ref in t_refs:

        sliceDf = bhv.time_slice(meso_sync_event, t_ref-pre, t_ref+post)
        frame_time.append(sliceDf['t'].values - t_ref) # relative to t_ref)
        frame_idx = sliceDf.index.values # get frames

        N_mean.append(np.mean(N[:,frame_idx], axis = 0)) # Avg across neurons

    N_mean_time = bhv.tolerant_mean(N_mean)
    time = np.arange(-pre,post,(post+pre)/5)
    axes[i].plot(time,N_mean_time,'o-', color = 'black')

    axes[i].set_ylim([0,0.03])
    axes[i].set_title(event)
    axes[i].set_xlabel('Time (ms)')
    axes[i].set_ylabel('\u0394F/F')

fig.suptitle('Neuronal activity avg across neurons aligned to')
fig.tight_layout()

# %%
