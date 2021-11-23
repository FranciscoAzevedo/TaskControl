# %% imports 
%matplotlib qt5
%matplotlib qt5
%load_ext autoreload
%autoreload 2

# plotting
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

# Math
import scipy as sp
import scipy.signal
import numpy as np
import pandas as pd

# Misc
import calendar
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
# %%
animals_dir = Path('/media/storage/shared-paton/georg/Animals_reaching/')
animals = [ 'JJP-02909_Lifeguard','JJP-02911_Lumberjack','JJP-02912_Teacher','JJP-02994_Plumber',
            'JJP-02995_Poolboy','JJP-02996_Policeman','JJP-02997_Therapist']

animals_dir_plots = animals_dir / 'plots'
os.makedirs(animals_dir_plots, exist_ok=True)

no_sessions_to_analyze = 10

n_rews_collected = np.zeros((len(animals), no_sessions_to_analyze))
n_anticipatory = np.zeros((len(animals), no_sessions_to_analyze))
no_trials = np.zeros((len(animals), no_sessions_to_analyze))
session_length = np.zeros((len(animals), no_sessions_to_analyze)) 

for j, animal in enumerate(animals):

    SessionsDf = utils.get_sessions(animals_dir / animal)
    
    # Filter sessions to the ones of the task we want to see
    task_name = ['learn_to_choose_v2']
    FilteredSessionsDf = pd.concat([SessionsDf.groupby('task').get_group(name) for name in task_name])
    log_paths = [Path(path)/'arduino_log.txt' for path in FilteredSessionsDf['path']]

    for k,log_path in enumerate(tqdm(log_paths[:no_sessions_to_analyze], position=0, leave=True, desc=animal)):
        
        LogDf = bhv.get_LogDf_from_path(log_path)

        # Getting metrics
        TrialSpans = bhv.get_spans_from_names(LogDf, "TRIAL_ENTRY_STATE", "ITI_STATE")

        TrialDfs = []
        for i, row in TrialSpans.iterrows():
            TrialDfs.append(bhv.time_slice(LogDf, row['t_on'], row['t_off']))

        metrics = ( met.get_start, met.get_stop, met.get_correct_side, met.get_outcome, met.get_chosen_side, met.has_choice,
                    met.rew_collected)
        SessionDf = bhv.parse_trials(TrialDfs, metrics)

        # Session metrics
        n_rews_collected[j,k] = SessionDf['rew_collect'].sum()
        n_anticipatory[j,k] = SessionDf['has_choice'].sum()

        no_trials[j,k] = len(SessionDf)
        session_length [j,k] = (LogDf['t'].iloc[-1]-LogDf['t'].iloc[0])/(1000*60) # convert msec. -> sec.-> min.

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

# %% Reward rate, perc. of anticipatory trials and rewardss collected
fig , axes = plt.subplots(ncols=3, figsize=(10,4))

for i,animal in enumerate(animals):
    # Overal reward rate
    axes[0].plot(n_rews_collected[i,:]/session_length[i,:],alpha = 0.5, label = animal)

    # Perc of anticipatory
    axes[1].plot(n_anticipatory[i,:]/no_trials[i,:],alpha = 0.5, label = animal)

    # Perc of rew collected not normed by time
    axes[2].plot(n_rews_collected[i,:]/no_trials[i,:],alpha = 0.5, label = animal)

# Mean traces
axes[0].plot(np.mean(n_rews_collected/session_length, axis = 0), color = 'k', label = 'mean')

axes[1].plot(np.mean(n_anticipatory/no_trials, axis = 0), color = 'k', label = 'mean')

axes[2].plot(np.mean(n_rews_collected/no_trials, axis = 0), color = 'k', label = 'mean')

for ax in axes:
    ax.set_xlabel('Session no.')

axes[0].set_ylabel('Rewards per minute')
axes[1].set_ylabel('Perc. anticipatory')
axes[2].set_ylabel('Perc. rewards collected')
axes[2].legend(loc='upper left',frameon=False, fontsize = 'xx-small')

fig.tight_layout()
plt.savefig(animals_dir_plots / ('initial_training_metrics.png'), dpi=600)

# %%
