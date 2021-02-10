# %%
%matplotlib qt5
%load_ext autoreload
%autoreload 2

from matplotlib import pyplot as plt
import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 331
mpl.rcParams['figure.dpi'] = 166 # the screens in the viv
import behavior_analysis_utils as bhv
import pandas as pd

# this should be changed ... 
from pathlib import Path
import scipy as sp
import numpy as np
import seaborn as sns
from tqdm import tqdm
import os
import utils

from behavior_plotters import *

# %% Preprocessing: LC syncing
log_path = utils.get_file_dialog()

plot_dir = log_path.parent / 'plots'
os.makedirs(plot_dir, exist_ok=True)

# %%
LoadCellDf, t_harp = bhv.parse_bonsai_LoadCellData(log_path.parent / "bonsai_LoadCellData.csv")

arduino_sync = bhv.get_arduino_sync(log_path, sync_event_name="TRIAL_ENTRY_EVENT")
t_harp = t_harp['t'].values
t_arduino = arduino_sync['t'].values
# pd.read_csv(log_path.parent / "arduino_sync.csv")['t'].values

if t_harp.shape != t_arduino.shape:
    t_arduino, t_harp = bhv.cut_timestamps(t_arduino, t_harp, verbose = True)

m, b = bhv.sync_clocks(t_harp, t_arduino, log_path)
LogDf = pd.read_csv(log_path.parent / "LogDf.csv")

# %%
TrialSpans = bhv.get_spans_from_names(LogDf, "TRIAL_ENTRY_STATE", "ITI_STATE")

TrialDfs = []
for i, row in tqdm(TrialSpans.iterrows(), position=0, leave=True):
    # print(bhv.time_slice(LogDf, row['t_on'], row['t_off'], mode='inclusive'))
    TrialDfs.append(bhv.time_slice(LogDf, row['t_on'], row['t_off'], mode='inclusive'))

metrics = (bhv.get_start, bhv.get_stop, bhv.has_choice, bhv.get_choice, bhv.choice_RT, bhv.get_in_corr_loop, \
                bhv.is_successful, bhv.get_outcome, bhv.get_instructed, bhv.get_bias, bhv.get_correct_zone, bhv.get_x_thresh)

SessionDf = bhv.parse_trials(TrialDfs, metrics)
t_start = SessionDf['t_on'].iloc[0]
t_stop = SessionDf['t_off'].iloc[-1]

LCDf = bhv.time_slice(LoadCellDf, t_start, t_stop)


# %%
