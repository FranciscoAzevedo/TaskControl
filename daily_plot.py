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
import calendar

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
session_metrics = ( met.get_start, met.get_stop, met.get_correct_side, met.get_interval_category, met.get_outcome, 
            met.get_chosen_side, met.has_reach_left, met.has_reach_right, met.get_in_corr_loop,  
            met.reach_rt_left, met.reach_rt_right, met.has_choice, met.get_interval, met.get_timing_trial,
            met.get_choice_rt, met.get_reached_side, met.get_bias, met.get_init_rt, met.rew_collected)

# CHANGE IN CASE SESSIONS DONT HAVE TRIAL INIT
SessionDf, TrialDfs = utils.get_SessionDf(LogDf, session_metrics, "TRIAL_AVAILABLE_EVENT", "ITI_STATE")

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

"""
..######..####.##....##..######...##.......########
.##....##..##..###...##.##....##..##.......##......
.##........##..####..##.##........##.......##......
..######...##..##.##.##.##...####.##.......######..
.......##..##..##..####.##....##..##.......##......
.##....##..##..##...###.##....##..##.......##......
..######..####.##....##..######...########.########
"""
# Session overview with outcome on background
fig, axes = plt.subplots(figsize=[10,2])

bhv_plt_reach.plot_session_overview(SessionDf, animal_meta, session_date, axes = axes)
plt.savefig(plot_dir / ('session_overview.png'), dpi=600)

# 1st reach choice RT
choice_interval = 2000 # ms
bin_width = 100 # ms

bhv_plt_reach.plot_choice_RT_hist(SessionDf, choice_interval, bin_width)
plt.savefig(plot_dir / ('choice_RTs.png'), dpi=600)

try:
    # Grasp duration distro split by outcome and choice
    bin_width = 5 # ms
    max_reach_dur = 250 # ms

    perc = 25 #th 

    bhv_plt_reach.plot_grasp_duration_distro(LogDf, SessionDf, bin_width, max_reach_dur, perc)
    plt.savefig(plot_dir / ('hist_grasp_dur_' + str(perc) + '_percentile.png'), dpi=600)
except:
    print('Couldnt run Grasp dur plot')

try:
    # Histogram of number of reaches per trial 
    reach_crop = 10

    bhv_plt_reach.plot_hist_no_reaches_per_trial(LogDf, reach_crop)
    plt.savefig(plot_dir / ('hist_no_reaches_per_trial.png'), dpi=600)
except:
    print('Couldnt run no reaches per trial plot')

#  Trial initiation distro
colors = dict(  correct="#72E043", incorrect="#F56057", premature="#9D5DF0", 
                missed="#F7D379", jackpot='#f05dc9', anticipatory="#CB9F45")

bhv_plt_reach.plot_init_times(SessionDf, colors)

plt.savefig(plot_dir / ('trial_init_rt_scatter.png'), dpi=600)

#  Trials initiated moving average within session
window_sizes = [5,10] # min
ylims = [0,6]
fig, axes = plt.subplots(figsize=(5, 4))

init_times = (np.array(SessionDf['t_on'].values)-LogDf['t'].iloc[0])/(1000*60) # msec-> sec -> min
linspace = np.arange(0, round(max(init_times)))

n, edges = np.histogram(init_times, bins = linspace) # a bin every minute

# Uniform window to convolve every 
for window_size in window_sizes:
    y_mov_avg = np.convolve(n, np.ones(window_size)/window_size, mode='valid')
    axes.plot(y_mov_avg, label = str(window_size))

# Formatting
axes.axhline(5, linestyle=':', alpha = 0.5, color = 'k')
axes.set_xlabel('Time (min)')
axes.set_ylabel('Number of trials initiated')
axes.set_ylim(ylims)
axes.legend(loc='upper right', frameon=False, title = 'window (min)')
axes.set_title('Rolling mean of trial initiations')
plt.savefig(plot_dir / ('trial_init_rolling_mean.png'), dpi=600)

#  Plots no. of attempts before actual init over session
peaks_coincidence = 50 # how distant can two peaks be until they are not considered simultaneous anymore
hist_crop = 5

bhv_plt_reach.plot_no_init_attempts(LoadCellDf, TrialDfs, log_path, peaks_coincidence, hist_crop)

plt.savefig(plot_dir / ('attempts_before_init.png'), dpi=600)

plt.close('all')

"""
....###.....######..########...#######...######...######.
...##.##...##....##.##.....##.##.....##.##....##.##....##
..##...##..##.......##.....##.##.....##.##.......##......
.##.....##.##.......########..##.....##..######...######.
.#########.##.......##...##...##.....##.......##.......##
.##.....##.##....##.##....##..##.....##.##....##.##....##
.##.....##..######..##.....##..#######...######...######.
"""
# %% Obtain log_paths and plot dirs
animal_fd_path = Path(animal_fd_path)
across_session_plot_dir = animal_fd_path / 'plots'
animal_meta = pd.read_csv(animal_fd_path / 'animal_meta.csv')
nickname = animal_meta[animal_meta['name'] == 'Nickname']['value'].values[0]
os.makedirs(across_session_plot_dir, exist_ok=True)

# %% across sessions - plot weight
SessionsDf = utils.get_sessions(animal_fd_path)
Df = pd.read_csv(animal_fd_path / 'animal_meta.csv')
ini_weight = float(Df[Df['name'] == 'Weight']['value'])

for i,row in SessionsDf.iterrows():
    try:
        path = row['path']
        Df = pd.read_csv(Path(path) / 'animal_meta.csv')
        current_weight = float(Df[Df['name'] == 'current_weight']['value'])
        SessionsDf.loc[row.name,'weight'] = current_weight
        SessionsDf.loc[row.name,'weight_frac'] = current_weight / ini_weight
    except:
        pass

# Formatting
fig, axes = plt.subplots()
axes.plot(SessionsDf.index.values,SessionsDf.weight_frac,'o')
axes.set_xticks(SessionsDf.index.values)
axes.set_xticklabels(SessionsDf['date'].values,rotation=90)
line_kwargs = dict(lw=1,linestyle=':',alpha=0.75)
axes.axhline(0.85, color='g', **line_kwargs)
axes.axhline(0.75, color='r', **line_kwargs)
axes.set_ylim(0.5,1)
axes.set_title('Weight across sessions (%s)' %nickname)
axes.set_xlabel('session date')
axes.set_ylabel('weight (%)')
fig.tight_layout()
plt.savefig(across_session_plot_dir / ('weight_across_sessions.png'), dpi=600)

# %% Evolution of trial outcome 
SessionsDf = utils.get_sessions(animal_fd_path)
    
# Filter sessions to the ones of the task we want to see
task_name = ['learn_to_choose_v2']
FilteredSessionsDf = pd.concat([SessionsDf.groupby('task').get_group(name) for name in task_name])
log_paths = [Path(path)/'arduino_log.txt' for path in FilteredSessionsDf['path']]

# Obtain the perc of reaches, correct and incorrect trials
perc_corr_left, perc_corr_right, perc_correct, perc_pre = [],[],[],[]
perc_missed, perc_missed_left, perc_missed_right = [],[],[]
date_abbr,no_trials,session_length, mondays = [],[],[],[]
n_rewards = []

for i,log_path in enumerate(log_paths):
    
    path = log_path.parent 
    LogDf = bhv.get_LogDf_from_path(log_path)

    # Correct date format
    folder_name = os.path.basename(path)
    date_str = folder_name.split('_')[0].split('-')
    date = [int(d) for d in date_str]

    # Vertical lines on monday
    weekday = calendar.weekday(date[0],date[1],date[2])
    if weekday == 0:
        mondays.append(i)

    month_abbr = calendar.month_abbr[date[1]]
    date_abbr.append(month_abbr+'-'+str(date[2]))

    # Getting metrics
    TrialSpans = bhv.get_spans_from_names(LogDf, "TRIAL_ENTRY_STATE", "ITI_STATE")

    TrialDfs = []
    for i, row in tqdm(TrialSpans.iterrows(),position=0, leave=True):
        TrialDfs.append(bhv.time_slice(LogDf, row['t_on'], row['t_off']))

    metrics = (met.get_start, met.get_stop, met.get_correct_side, met.get_outcome, met.get_chosen_side, met.has_choice)
    SessionDf = bhv.parse_trials(TrialDfs, metrics)

    # Session metrics
    MissedDf = SessionDf[SessionDf['outcome'] == 'missed']
    choiceDf = SessionDf[SessionDf['has_choice'] == True]
    left_trials_missedDf = bhv.groupby_dict(SessionDf, dict(outcome='missed', correct_side='left'))
    right_trials_missedDf = bhv.groupby_dict(SessionDf, dict(outcome='missed', correct_side='right'))

    corr_leftDf = bhv.groupby_dict(SessionDf, dict(outcome='correct', correct_side='left'))
    left_trials_with_choiceDf = bhv.groupby_dict(SessionDf, dict(has_choice=True, correct_side='left'))
    corr_rightDf = bhv.groupby_dict(SessionDf, dict(outcome='correct', correct_side='right'))
    right_trials_with_choiceDf = bhv.groupby_dict(SessionDf, dict(has_choice=True, correct_side='right'))

    # Metrics of evolution
    try:
        perc_corr_left.append(len(corr_leftDf)/len(left_trials_with_choiceDf)*100)
    except:
        perc_corr_left.append(np.nan)

    try:
        perc_corr_right.append(len(corr_rightDf)/len(right_trials_with_choiceDf)*100)
    except:
        perc_corr_right.append(np.nan)

    perc_correct.append((SessionDf.outcome == 'correct').sum()/len(choiceDf)*100)

    perc_missed.append(len(MissedDf)/len(SessionDf)*100) # exclude jackpot rews
    perc_missed_left.append(len(left_trials_missedDf)/len(SessionDf[SessionDf['correct_side'] == 'left'])*100)
    perc_missed_right.append(len(right_trials_missedDf)/len(SessionDf[SessionDf['correct_side'] == 'right'])*100)

    if len(SessionDf[SessionDf.outcome == 'premature']) != 0:
        perc_pre.append(sum(SessionDf.outcome == 'premature')/len(SessionDf)*100)
    else:
        perc_pre.append(0)

    n_rewards.append(len(bhv.get_events_from_name(LogDf,'REWARD_STATE')))

    no_trials.append(len(SessionDf))
    session_length.append((LogDf['t'].iloc[-1]-LogDf['t'].iloc[0])/(1000*60)) # convert msec. -> sec.-> min.

# Plotting
fig , axes = plt.subplots(figsize=(10,4))

axes.plot(perc_corr_left, color = 'orange', label = 'Corr L (%)',alpha = 0.25)
axes.plot(perc_corr_right, color = 'blue', label = 'Corr R (%)',alpha = 0.25)
axes.plot(perc_correct, color = 'green', label = 'Correct (%)', marker='o', markersize=2)

axes.plot(perc_missed_left, linestyle='dashed', color = 'orange', label = 'Missed L (%)',alpha = 0.25)
axes.plot(perc_missed_right, linestyle='dashed', color = 'blue', label = 'Missed R (%)',alpha = 0.25)
axes.plot(perc_missed, linestyle='dashed', color = 'grey', label = 'Missed (%)', marker='o', markersize=2)

axes.plot(perc_pre, color = 'pink', label = 'Premature (%)', marker='o', markersize=2)

[axes.axvline(monday, color = 'k', alpha = 0.25) for monday in mondays]

axes.set_title(' Session overview for ' + nickname + ' ' + str(task_name))
axes.set_ylabel('Trial outcome (%)')
axes.set_xlabel('Session date')
axes.legend(loc="center", frameon=False, bbox_to_anchor=(0.5, 0.98), prop={'size': 7}, ncol=7) 

plt.setp(axes, xticks=np.arange(0, len(date_abbr), 1), xticklabels=date_abbr)
plt.setp(axes, yticks=np.arange(0, 100+1, 10), yticklabels=np.arange(0, 100+1, 10))
plt.xticks(rotation=90)
fig.tight_layout()
plt.savefig(across_session_plot_dir / ('overview_across_sessions' + str(task_name)+ '.png'), dpi=600)

# %% Reward rate across sessions
fig , axes = plt.subplots(figsize=(10,4))

axes.plot(date_abbr, n_rewards/np.array(session_length))
axes.set_xlabel('Session date')
axes.set_ylabel('Rewards per minute')
plt.title('Reward rate across sessions')
plt.xticks(rotation=90)

[axes.axvline(monday, color = 'k', alpha = 0.25) for monday in mondays]

axes.set_ylim([0,6])
axes.set_title(' Session overview for ' + nickname + ' ' + str(task_name))
fig.tight_layout()
plt.savefig(across_session_plot_dir / ('rew_rate_across_sessions.png'), dpi=600)

# %% Reaches during delay across sessions
tasks_names = ['learn_to_choose_v2']
init_day_idx = 0
bhv_plt_reach.reaches_during_delay_across_sess(animal_fd_path, tasks_names, init_day_idx)

plt.savefig(across_session_plot_dir / ('CDF_reach_during_delay_sessions.png'), dpi=600)
