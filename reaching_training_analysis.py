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

fd_path = utils.get_folder_dialog(initial_dir="/media/storage/shared-paton/georg/Animals_reaching/")

## DeepLabCut data
#h5_path = utils.get_file_dialog()
#DlcDf = read_dlc_h5(h5_path)
#bodyparts = sp.unique([j[0] for j in DlcDf.columns[1:]]) # all body parts

# Video data
#video_path = fd_path / "bonsai_video.avi"
#Vid = dlc_utils.read_video(str(video_path))

# Arduino data
log_path = fd_path / 'arduino_log.txt'
LogDf = bhv.get_LogDf_from_path(log_path)

# LoadCell data
LoadCellDf = bhv.parse_bonsai_LoadCellData(fd_path / 'bonsai_LoadCellData.csv')

## Synching 

from Utils import sync

# Parse sync events/triggers
cam_sync_event = sync.parse_cam_sync(fd_path / 'bonsai_frame_stamps.csv')
lc_sync_event = sync.parse_harp_sync(fd_path / 'bonsai_harp_sync.csv')
arduino_sync_event = sync.get_arduino_sync(fd_path / 'arduino_log.txt')

# Get the values out of them
Sync = sync.Syncer()
Sync.data['arduino'] = arduino_sync_event['t'].values
Sync.data['loadcell'] = lc_sync_event['t'].values
#Sync.data['dlc'] = cam_sync_event.index.values # the frames are the DLC
#Sync.data['cam'] = cam_sync_event['t'].values # used for what?

# Sync them all to master clock (arduino [FSM?] at ~1Khz)
LoadCellDf['t_loadcell'] = LoadCellDf['t'] # keeps the original
Sync.sync('loadcell','arduino')
LoadCellDf['t'] = Sync.convert(LoadCellDf['t'].values, 'loadcell', 'arduino')

# Transform the time of the DLC into arduino time
#DlcDf['t'] = Sync.convert(DlcDf.index.values, 'dlc', 'arduino')

## SessionDf and Go cue

# Add single GO_CUE_EVENT
LogDf = bhv.add_go_cue_LogDf(LogDf)

session_metrics = ( met.get_start, met.get_stop, met.get_correct_side, met.get_interval_category, met.get_outcome, 
            met.get_chosen_side, met.has_reach_left, met.has_reach_right, met.get_in_corr_loop,
            met.reach_rt_left, met.reach_rt_right, met.has_choice, met.get_interval, met.get_timing_trial,
            met.get_choice_rt, met.get_reached_side, met.get_bias, met.get_init_rt, met.rew_collected) 

SessionDf, TrialDfs = utils.get_SessionDf(LogDf, session_metrics, "TRIAL_ENTRY_EVENT", "ITI_STATE")

# Moving average mean subtraction (need to do after synching)
samples = 1000 # ms
LoadCellDf['x'] = LoadCellDf['x'] - LoadCellDf['x'].rolling(samples).mean()
LoadCellDf['y'] = LoadCellDf['y'] - LoadCellDf['y'].rolling(samples).mean()

# Create boolean vars for each outcome 
outcomes = SessionDf['outcome'].unique()
for outcome in outcomes:
   SessionDf['is_'+outcome] = SessionDf['outcome'] == outcome

#  Plots dir and animal info
animal_meta = pd.read_csv(log_path.parent.parent / 'animal_meta.csv')
nickname = animal_meta[animal_meta['name'] == 'Nickname']['value'].values[0]
session_date = log_path.parent.stem.split('_')[0]

plot_dir = log_path.parent / 'plots'
os.makedirs(plot_dir, exist_ok=True)

# %% defining some stuff
Skeleton   = (('D1L','J1L'),('D2L','J2L'),('D3L','J3L'),('D4L','J4L'),('D5L','J5L'),
             ('PR','J1R'),('PR','J2R'),('PR','J3R'),('PR','J4R'),('PR','J5R'),
             ('D1R','J1R'),('D2R','J2R'),('D3R','J3R'),('D4R','J4R'),('D5R','J5R'),
             ('PL','J1L'),('PL','J2L'),('PL','J3L'),('PL','J4L'),('PL','J5L'))

paws = ['PL','PR']


"""
 #       #######    #    ######  #     #    ####### #######    ######  #######    #     #####  #     #
 #       #         # #   #     # ##    #       #    #     #    #     # #         # #   #     # #     #
 #       #        #   #  #     # # #   #       #    #     #    #     # #        #   #  #       #     #
 #       #####   #     # ######  #  #  #       #    #     #    ######  #####   #     # #       #######
 #       #       ####### #   #   #   # #       #    #     #    #   #   #       ####### #       #     #
 #       #       #     # #    #  #    ##       #    #     #    #    #  #       #     # #     # #     #
 ####### ####### #     # #     # #     #       #    #######    #     # ####### #     #  #####  #     #

"""

# %% Do they react (fast) to valve opening? Or anticipate it?
align_events = ['REWARD_LEFT_VALVE_ON','REWARD_RIGHT_VALVE_ON']
pre,post = 500,3000 # ms

bin_width = 100 # ms

fig, axes = plt.subplots(nrows = 2, figsize=(4, 5))

for align_event,axis in zip(align_events,axes):
    bhv_plt_reach.plot_reaches_window_aligned_on_event(LogDf, align_event, pre, post, bin_width,axes = axis)

fig.suptitle('Reaches aligned to')
fig.tight_layout()
plt.savefig(plot_dir / ('hist_of_reaches_aligned_to_valve_opening.png'), dpi=600)

# %% Percentage of anticipatory reaches
n_anticipatory = round(len(SessionDf[SessionDf['outcome'] == 'anticipatory'])/len(SessionDf)*100, 1)

print('\n Performing ' + str(n_anticipatory) + "% anticipatory reaches")

# %%

"""
 #       #######    #    ######  #     #    ####### #######     #####  #     # ####### #######  #####  #######
 #       #         # #   #     # ##    #       #    #     #    #     # #     # #     # #     # #     # #
 #       #        #   #  #     # # #   #       #    #     #    #       #     # #     # #     # #       #
 #       #####   #     # ######  #  #  #       #    #     #    #       ####### #     # #     #  #####  #####
 #       #       ####### #   #   #   # #       #    #     #    #       #     # #     # #     #       # #
 #       #       #     # #    #  #    ##       #    #     #    #     # #     # #     # #     # #     # #
 ####### ####### #     # #     # #     #       #    #######     #####  #     # ####### #######  #####  #######

"""

# %% 1st reach choice RT
choice_interval = 3000 # ms
bin_width = 100 # ms

bhv_plt_reach.plot_choice_RT_hist(SessionDf, choice_interval, bin_width)
plt.savefig(plot_dir / ('choice_RTs.png'), dpi=600)

# %% Grasp duration distro split by outcome and choice
bin_width = 5 # ms
max_grasp_dur = 250 # ms

perc = 25 #th 

bhv_plt_reach.plot_grasp_duration_distro(LogDf, SessionDf, bin_width, max_grasp_dur, perc)
plt.savefig(plot_dir / ('hist_grasp_dur_' + str(perc) + '_percentile.png'), dpi=600)

# %% Histogram of number of reaches per trial 
reach_crop = 12

bhv_plt_reach.plot_hist_no_reaches_per_trial(LogDf, reach_crop)
plt.savefig(plot_dir / ('hist_no_reaches_per_trial.png'), dpi=600)

# %% Are they using a sampling strategy? 
fig, axes = plt.subplots(figsize=(3, 4))

missesDf_idx = SessionDf['outcome'] == 'missed'
choiceDf = SessionDf[~missesDf_idx] # drop rows with missed trials

# What is the prob if going right first in left side trials
left_sideDf = choiceDf[choiceDf['correct_side'] == 'left']
try:
    reached_sideDf = choiceDf.groupby(['correct_side','reached_side']).get_group(('left','both'))
    perc_l_after_r = len(reached_sideDf)/len(left_sideDf)*100
except:
    perc_l_after_r = 0

# And the inverse here
right_sideDf = choiceDf[choiceDf['correct_side'] == 'right']
try:
    reached_sideDf = choiceDf.groupby(['correct_side','reached_side']).get_group(('right','both'))
    perc_r_after_l = len(reached_sideDf)/len(right_sideDf)*100
except:
    perc_r_after_l = 0

labels = ['L after R', 'R after L']

# Plotting groups
axes.bar(labels, [perc_l_after_r, perc_r_after_l])

axes.set_title('Prob of going X after Y')
axes.set_ylabel('Prob. (%)')
axes.set_ylim([0,75])
axes.legend(loc='upper left', frameon=False) 

fig.tight_layout()

plt.savefig(plot_dir / ('prob_X_after_Y.png'), dpi=600)

# %% How many trials are they actually getting because they sample?
correct_Df = SessionDf[SessionDf['outcome'] == 'correct']

# trials labelled as correct but where the first reach does not match the correct side
# are sampled trials because subsequent reaches after the first get the reward
sampled_trialsDf = correct_Df[correct_Df['correct_side'] != correct_Df['chosen_side']]
sampled_frac = round(len(sampled_trialsDf)/len(correct_Df)*100,1)
sampled_frac_session = round(len(sampled_trialsDf)/len(SessionDf)*100,1) # against whole session

print('Perc of sampled trials against correct: ' +str(sampled_frac)+ '% \n And against all: '+ str(sampled_frac_session) + '%')

# %% In how many trials do they miss because they have reaches with insufficient duration?
# Doesnt desambiguate which reach it was, just that there was a reach that was shorter than needed

missedDf = SessionDf[SessionDf['outcome'] == 'missed']

# Not considering when they try both sides!
correct_attemptsDf = missedDf[missedDf['correct_side'] == missedDf['reached_side']]

# Considering they try to both
both_side_attemptsDf = missedDf[missedDf['reached_side'] == 'both']

# In how many trials are there attempts below duration threshold
corr_attempts_perc = round(len(correct_attemptsDf)/len(SessionDf)*100,2)

# Or attempts to both sides
both_side_perc = round(len(both_side_attemptsDf)/len(SessionDf)*100,2)

print('There were ' + str(corr_attempts_perc)+ '% (putatively correct) trials with reach durations below threshold')
print('And ' + str(both_side_perc) + '% with reaches for both sides')

# %% What happens the trial after they attempt to reach but the grasp is below threshold duration
fig, axes = plt.subplots()

# Get Df of trials after attempt
corr_attempts_idx = correct_attemptsDf.index
trials_after_attemptDf = SessionDf.iloc[corr_attempts_idx+1]

colors = dict(correct="#72E043", incorrect="#F56057", premature="#9D5DF0", missed="#F7D379", jackpot ='#FFC0CB')
bar_width = 0.20

Dfs = [trials_after_attemptDf, SessionDf]

outcomes = SessionDf['outcome'].unique()
labels = ['after_attempt', 'session']

# get fraction of each outcome relative to total Session size
bar_frac = np.zeros([len(Dfs),len(outcomes)])

for i,Df in enumerate(Dfs): 
    for j,outcome in enumerate(outcomes):
        try:
            # number of trials with outcome 
            outcome_trials = len(Df.groupby(['outcome']).get_group(outcome))
            # normalize for session len
            bar_frac[i,j] = outcome_trials/len(Df) 
        except:
            bar_frac[i,j] = 0 # when there are no trials

# Plotting
y_offset =  np.zeros(len(Dfs))

# For each outcome
for col,outcome in enumerate(outcomes):
    plt.bar(labels, bar_frac[:,col], bar_width, bottom=y_offset, label = outcome, color=colors[outcome])
    y_offset = y_offset + bar_frac[:,col]

for label,Df in zip(labels, Dfs):
    axes.text(label, 0.9, 'n='+str(len(Df)))

axes.set_title('Percentage of trial outcome after attempt and overall session')
axes.legend(loc="center", frameon = False, bbox_to_anchor=(0.5, 1.1), ncol=len(colors))
axes.set_ylabel('Fraction of trials')
axes.set_ylim([0,1])

# %%

"""
 #       #######    #    ######  #     #    ####### #######    ### #     # ### #######
 #       #         # #   #     # ##    #       #    #     #     #  ##    #  #     #
 #       #        #   #  #     # # #   #       #    #     #     #  # #   #  #     #
 #       #####   #     # ######  #  #  #       #    #     #     #  #  #  #  #     #
 #       #       ####### #   #   #   # #       #    #     #     #  #   # #  #     #
 #       #       #     # #    #  #    ##       #    #     #     #  #    ##  #     #
 ####### ####### #     # #     # #     #       #    #######    ### #     # ###    #

"""

# %% Reaching CDF's for short and long trials (useful before learn to time)
fig, axes = plt.subplots(ncols = 2, figsize=(6, 3))

bhv_plt_reach.CDF_of_reaches_during_delay(SessionDf,TrialDfs, axes = axes)
fig.suptitle('CDF of first reach split on trial type')
fig.tight_layout()
plt.savefig(plot_dir / ('CDF_of_reaches_during_delay.png'), dpi=600)

# %% Trials initiated moving average within session
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

# %% How hard do they push to initiate trials?
window = 300 # ms
ylims = [-2000,2000]
align_event = 'TRIAL_ENTRY_EVENT'
fig, axes = plt.subplots(ncols=2,sharex=True, figsize=(6,3))

linspace = np.linspace(-window, window, 2*window).T

X,Y = bhv_plt_reach.get_LC_slice_aligned_on_event(LoadCellDf, TrialDfs, align_event, window, window) 

X = bhv_plt_reach.truncate_pad_vector(X)
Y = bhv_plt_reach.truncate_pad_vector(Y)

# Average traces
X_mean = np.mean(X, axis=0)
Y_mean = np.mean(Y, axis=0)

axes[0].plot(linspace, X.T, alpha=0.25, c = 'tab:orange')
axes[0].plot(linspace, X_mean, c = 'k')
axes[1].plot(linspace, Y.T, alpha=0.25, c = 'tab:blue')
axes[1].plot(linspace, Y_mean, c = 'k')

# Formatting
axes[0].set_ylabel('Force (a.u.)')

for ax in axes:
    ax.set_ylim(ylims)
    ax.axvline(0, linestyle='dotted', color='k', alpha = 0.5)
    ax.set_xlabel('Time (ms)')

fig.suptitle('LC forces aligned to ' + str(align_event))
fig.tight_layout()
plt.savefig(plot_dir / ('trial_init_forces.png'), dpi=600)

# %% Plots no. of attempts before actual init over session
init_tresh = 300

bhv_plt_reach.plot_no_init_attempts(LoadCellDf, TrialDfs, init_tresh)

plt.savefig(plot_dir / ('attempts_before_init.png'), dpi=600)

# %% Trial initiation distro
colors = dict(correct="#72E043", incorrect="#F56057", premature="#9D5DF0", missed="#F7D379", jackpot='#f05dc9')

bhv_plt_reach.plot_init_times(SessionDf, colors)

plt.savefig(plot_dir / ('trial_init_rt_scatter.png'), dpi=600)

# %%
"""
 #       #######    #    ######  #     #    ####### #######    ####### ### #     # #######
 #       #         # #   #     # ##    #       #    #     #       #     #  ##   ## #
 #       #        #   #  #     # # #   #       #    #     #       #     #  # # # # #
 #       #####   #     # ######  #  #  #       #    #     #       #     #  #  #  # #####
 #       #       ####### #   #   #   # #       #    #     #       #     #  #     # #
 #       #       #     # #    #  #    ##       #    #     #       #     #  #     # #
 ####### ####### #     # #     # #     #       #    #######       #    ### #     # #######

"""
# %% Session overview simple
pre,post = 500,5000
align_event = 'PRESENT_INTERVAL_STATE'

bhv_plt_reach.plot_overview_simple(LogDf, align_event, pre, post)
plt.savefig(plot_dir / ('session_overview_simple.png'), dpi=600)

# %% Session overview w/reaches and aligned to 1st and 2nd cues, split by outcome
pre,post = 500,4000
align_event = 'PRESENT_INTERVAL_STATE'

bhv_plt_reach.plot_overview_aligned_1st_2nd(TrialDfs, SessionDf, pre, post)
plt.savefig(plot_dir / ('session_overview_aligned_on_1st_2nd.png'), dpi=600)

# %% Session overview with outcome on background
fig, axes = plt.subplots(figsize=[10,2])

bhv_plt_reach.plot_session_overview(SessionDf, animal_meta, session_date, axes = axes)
plt.savefig(plot_dir / ('session_overview.png'), dpi=600)

# %% outcome_split_by_interval_category
bhv_plt_reach.outcome_split_by_interval_category(SessionDf)
plt.savefig(plot_dir / ('outcome_split_by_interval_category.png'), dpi=600)

# %% Psychometric timing trials

# filter out jackpot
jackpot_idx = SessionDf['outcome'] == 'jackpot'
cleanDf = SessionDf[~jackpot_idx]

# identify premature
pre_idx = cleanDf['outcome'] == 'premature'

bhv_plt_reach.plot_psychometric(cleanDf[~pre_idx], discrete=True)
plt.savefig(plot_dir / ('psychometric_timing.png'), dpi=600)

# Pseudo-Psychometric with premature timing trials
bhv_plt_reach.plot_psychometric(cleanDf[pre_idx], discrete=True)
plt.savefig(plot_dir / ('pseudo_psychometric_timing.png'), dpi=600)


"""
    #     #####  ######  #######  #####   #####      #####  #######  #####   #####  ### ####### #     #  #####
   # #   #     # #     # #     # #     # #     #    #     # #       #     # #     #  #  #     # ##    # #     #
  #   #  #       #     # #     # #       #          #       #       #       #        #  #     # # #   # #
 #     # #       ######  #     #  #####   #####      #####  #####    #####   #####   #  #     # #  #  #  #####
 ####### #       #   #   #     #       #       #          # #             #       #  #  #     # #   # #       #
 #     # #     # #    #  #     # #     # #     #    #     # #       #     # #     #  #  #     # #    ## #     #
 #     #  #####  #     # #######  #####   #####      #####  #######  #####   #####  ### ####### #     #  #####

"""

# %% Obtain log_paths and plot dirs
animal_fd_path = utils.get_folder_dialog(initial_dir="/media/storage/shared-paton/georg/Animals_reaching/")
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
        perc_corr_left.append(np.NaN)

    try:
        perc_corr_right.append(len(corr_rightDf)/len(right_trials_with_choiceDf)*100)
    except:
        perc_corr_right.append(np.NaN)

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

# %% Trial init day
trial_init_day = 'ago-25'
init_day_idx = [i for i,day in enumerate(date_abbr) if day == trial_init_day][0]

# %% Trials per minute
fig , axes = plt.subplots()

tpm = np.array(no_trials)/np.array(session_length)

axes.plot(tpm[init_day_idx:])

axes.set_ylim([0,5]) # A trial every 15s is the theoretical limit at chance level given our task settings
axes.set_title(' Session overview for ' + nickname + ' ' + str(task_name))
plt.title('No. TPM across sessions')
plt.setp(axes, xticks=np.arange(0,len(date_abbr[init_day_idx:]),1), xticklabels=date_abbr[init_day_idx:])
plt.xticks(rotation=45)

plt.savefig(across_session_plot_dir / ('tpm_across_sessions.png'), dpi=600)

# %% Reward rate across sessions
fig , axes = plt.subplots(figsize=(10,4))

axes.plot(date_abbr, n_rewards/np.array(session_length))
axes.set_xlabel('Session date')
axes.set_ylabel('Rewards per minute')
plt.title('Reward rate across sessions')
plt.xticks(rotation=90)

[axes.axvline(monday, color = 'k', alpha = 0.25) for monday in mondays]

axes.set_title(' Session overview for ' + nickname + ' ' + str(task_name))
fig.tight_layout()
plt.savefig(across_session_plot_dir / ('rew_rate_across_sessions.png'), dpi=600)

# %% Reaches during delay across sessions
tasks_names = ['learn_to_choose_v2']
bhv_plt_reach.reaches_during_delay_across_sess(animal_fd_path, tasks_names, init_day_idx)

plt.savefig(across_session_plot_dir / ('CDF_reach_during_delay_sessions.png'), dpi=600)

# %% Grasp duration across sessions
bhv_plt_reach.grasp_dur_across_sess(date_abbr)

plt.savefig(across_session_plot_dir / ('grasp_dur_sessions.png'), dpi=600)

"""
  #####  ####### ####### #     # ######
 #     # #          #    #     # #     #
 #       #          #    #     # #     #
  #####  #####      #    #     # ######
       # #          #    #     # #
 #     # #          #    #     # #
  #####  #######    #     #####  #

""" 

# %% See what is the impact of the buzzer frequency on LCs
window = 150 # ms around event
sides = ['left','right']

data_fd_path = utils.get_folder_dialog()

plot_dir = data_fd_path / 'plots'
os.makedirs(plot_dir, exist_ok=True)

SessionsDf = utils.get_sessions(data_fd_path)
paths = [Path(path) for path in SessionsDf['path']]

fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(6,6))
colors = sns.color_palette(palette='turbo', n_colors=len(paths))

# For every session
for color_no, path in enumerate(paths[-2:]):

    ## LOADING ##
    log_path = path /'arduino_log.txt'
    LogDf = bhv.get_LogDf_from_path(log_path)

    # LoadCell data
    LoadCellDf, t_harp = bhv.parse_bonsai_LoadCellData(path / "bonsai_LoadCellData.csv",trig_len=100, ttol=50)

    # Moving average mean subtraction
    samples = 1000 # ms
    LoadCellDf['x'] = LoadCellDf['x'] - LoadCellDf['x'].rolling(samples).mean()
    LoadCellDf['y'] = LoadCellDf['y'] - LoadCellDf['y'].rolling(samples).mean()

    # Synching arduino 
    arduino_sync = bhv.get_arduino_sync(log_path, sync_event_name="TRIAL_ENTRY_EVENT")
    t_harp = t_harp['t'].values
    t_arduino = arduino_sync['t'].values

    if t_harp.shape != t_arduino.shape:
        t_arduino, t_harp = bhv.cut_timestamps(t_arduino, t_harp, verbose = True)

    m3, b3 = bhv.sync_clocks(t_harp, t_arduino, log_path = log_path)
    LogDf = pd.read_csv(path / "LogDf.csv") # re-load the LogDf (to make sure we keep the original arduino clock)

    #  Create SessionDf - For LEARN_TO_CHOOSE onwards
    TrialSpans = bhv.get_spans_from_names(LogDf, "TRIAL_AVAILABLE_STATE", "ITI_STATE")

    TrialDfs = []
    for i, row in tqdm(TrialSpans.iterrows(),position=0, leave=True):
        TrialDfs.append(bhv.time_slice(LogDf, row['t_on'], row['t_off']))

    metrics = (met.get_start, met.get_stop, met.get_correct_side)
    SessionDf = bhv.parse_trials(TrialDfs, metrics)

    ## PLOTTING ##
    f = open(log_path, "r")
    freq_lines = [line for line in f if 'SET buzz_freq_sep' in line]
    freq_value = int(freq_lines[0][-4:-2]) #FIXME HARCODED

    # For short and long trials
    for i, side in enumerate(sides):

        go_cue = 'GO_CUE_' + side.upper() + '_EVENT'

        TrialDfs_filt = bhv_plt_reach.filter_trials_by(SessionDf, TrialDfs, dict(correct_side=side))

        X,Y = bhv_plt_reach.get_LC_slice_aligned_on_event(LoadCellDf, TrialDfs_filt, go_cue, window, window) 

        linspace = np.linspace(-window, window, 2*window).T

        # Average traces
        X_mean = np.mean(X, axis = 1)
        Y_mean = np.mean(Y, axis = 1)
        
        # Time domain
        #axes[i,0].plot(linspace, X_mean, c = colors[color_no], label = str(freq_value) + 'Hz', zorder = -color_no, linewidth = 0.5)
        #axes[i,1].plot(linspace, Y_mean, c = colors[color_no], zorder = -color_no, linewidth= 0.5)

        # TODO ALL OF THIS FREQ ANALISYS
        # Freq Domain
        _, _, spec_X = sp.signal.spectrogram(X_mean)
        _, _, spec_Y = sp.signal.spectrogram(Y_mean)

        axes[i,0].imshow(spec_X, aspect='auto', cmap='hot_r', origin='lower', zorder = -color_no, label = str(freq_value) + 'Hz')
        axes[i,1].imshow(spec_X, aspect='auto', cmap='hot_r', origin='lower', zorder = -color_no)

        axes[i,0].legend(frameon=False, loc = 'upper left', fontsize="small")

axes[0,0].set_title('Short trial')
axes[1,0].set_title('Long trial')
axes[1,0].set_xlabel('Left LC')
axes[1,1].set_xlabel('Right LC')
fig.suptitle('Impact of separability of buzzer freq. on LCs')
fig.tight_layout()

plt.savefig(plot_dir / ('impact_of_buzzer_freq_on_LCs.png'), dpi=600)

# %%
