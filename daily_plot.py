# plotting
import matplotlib.pyplot as plt

# Math
import scipy as sp
import numpy as np
import pandas as pd

# Misc
import os
import sys
from tqdm import tqdm
from pathlib import Path

# Custom
from Utils import behavior_analysis_utils as bhv
from Utils import metrics as met
from Utils import utils
from Utils import sync
import behav_plotters_reach as bhv_plt_reach

 
"""
 #       #######    #    ######  ### #     #  #####
 #       #     #   # #   #     #  #  ##    # #     #
 #       #     #  #   #  #     #  #  # #   # #
 #       #     # #     # #     #  #  #  #  # #  ####
 #       #     # ####### #     #  #  #   # # #     #
 #       #     # #     # #     #  #  #    ## #     #
 ####### ####### #     # ######  ### #     #  #####

"""
animal_fd_path = sys.argv[1]

fd_path = Path(utils.get_sessions(animal_fd_path).iloc[-1]['path']) # Get last session fd_path

# Arduino data
log_path = fd_path / 'arduino_log.txt'
LogDf = bhv.get_LogDf_from_path(log_path)

# LoadCell data
LoadCellDf = bhv.parse_bonsai_LoadCellData(fd_path / 'bonsai_LoadCellData.csv')

# Parse sync events/triggers
lc_sync_event = sync.parse_harp_sync(fd_path / 'bonsai_harp_sync.csv')
arduino_sync_event = sync.get_arduino_sync(fd_path / 'arduino_log.txt')

# Get the values out of them
Sync = sync.Syncer()
Sync.data['arduino'] = arduino_sync_event['t'].values
Sync.data['loadcell'] = lc_sync_event['t'].values

# Add single GO_CUE_EVENT
LogDf = bhv.add_go_cue_LogDf(LogDf)

#  Create SessionDf 
TrialSpans = bhv.get_spans_from_names(LogDf, "TRIAL_AVAILABLE_STATE", "ITI_STATE")

TrialDfs = []
for i, row in tqdm(TrialSpans.iterrows(),position=0, leave=True):
    TrialDfs.append(bhv.time_slice(LogDf, row['t_on'], row['t_off']))

metrics = ( met.get_start, met.get_stop, met.get_correct_side, met.get_interval_category, met.get_outcome, 
            met.get_chosen_side, met.has_reach_left, met.has_reach_right, met.get_in_corr_loop,  
            met.reach_rt_left, met.reach_rt_right, met.has_choice, met.get_interval, met.get_timing_trial,
            met.get_choice_rt, met.get_reached_side, met.get_bias, met.is_anticipatory, met.get_init_rt) 

SessionDf = bhv.parse_trials(TrialDfs, metrics)

# Add choice grasp dur metric computed differently from the other metrics
SessionDf = bhv_plt_reach.compute_choice_grasp_dur(LogDf,SessionDf)

# Create boolean vars for each outcome 
outcomes_raw = SessionDf['outcome'].unique()
outcomes = [outcome_raw for outcome_raw in outcomes_raw if isinstance(outcome_raw, str)]
for outcome in outcomes:
   SessionDf['is_'+outcome] = SessionDf['outcome'] == outcome

# Plots dir and animal info
animal_meta = pd.read_csv(log_path.parent.parent / 'animal_meta.csv')
nickname = animal_meta[animal_meta['name'] == 'Nickname']['value'].values[0]
session_date = log_path.parent.stem.split('_')[0]

plot_dir = log_path.parent / 'plots'
os.makedirs(plot_dir, exist_ok=True)

# 1st reach choice RT
choice_interval = 3000 # ms
bin_width = 100 # ms

bhv_plt_reach.plot_choice_RT_hist(SessionDf, choice_interval, bin_width)
plt.savefig(plot_dir / ('choice_RTs.png'), dpi=600)

# Grasp duration distro split by outcome and choice
bin_width = 5 # ms
max_reach_dur = 250 # ms

perc = 25 #th 

bhv_plt_reach.plot_grasp_duration_distro(LogDf, SessionDf, bin_width, max_reach_dur, perc)
plt.savefig(plot_dir / ('hist_grasp_dur_' + str(perc) + '_percentile.png'), dpi=600)

# Histogram of number of reaches per trial 
reach_crop = 12

bhv_plt_reach.plot_hist_no_reaches_per_trial(LogDf, reach_crop)
plt.savefig(plot_dir / ('hist_no_reaches_per_trial.png'), dpi=600)

# Session overview with outcome on background
fig, axes = plt.subplots(figsize=[10,2])

bhv_plt_reach.plot_session_overview(SessionDf, animal_meta, session_date, axes = axes)
plt.savefig(plot_dir / ('session_overview.png'), dpi=600)

plt.close('all')
