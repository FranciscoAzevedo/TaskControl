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

## Video data
#video_path = fd_path / "bonsai_video.avi"
#Vid = read_video(str(video_path))

# Arduino data
log_path = fd_path / 'arduino_log.txt'
LogDf = bhv.get_LogDf_from_path(log_path)

# LoadCell data
LoadCellDf = bhv.parse_bonsai_LoadCellData(fd_path / 'bonsai_LoadCellData.csv')

# Moving average mean subtraction
samples = 1000 # ms
LoadCellDf['x'] = LoadCellDf['x'] - LoadCellDf['x'].rolling(samples).mean()
LoadCellDf['y'] = LoadCellDf['y'] - LoadCellDf['y'].rolling(samples).mean()

# %%  Synching 

from Utils import sync
#cam_sync_event = sync.parse_cam_sync(fd_path / 'bonsai_frame_stamps.csv')
lc_sync_event = sync.parse_harp_sync(fd_path / 'bonsai_harp_sync.csv')
arduino_sync_event = sync.get_arduino_sync(fd_path / 'arduino_log.txt')

Sync = sync.Syncer()
Sync.data['arduino'] = arduino_sync_event['t'].values
Sync.data['loadcell'] = lc_sync_event['t'].values
#Sync.data['dlc'] = cam_sync_event.index.values # the frames are the DLC
#Sync.data['cam'] = cam_sync_event['t'].values # used for what?
Sync.sync('arduino','loadcell')

#DlcDf['t'] = Sync.convert(DlcDf.index.values, 'dlc', 'arduino')

# %% SessionDf and Go Cue

# ADD SINGLE GO_CUE_EVENT
LogDf = bhv.add_go_cue_LogDf(LogDf)

#  Create SessionDf - For LEARN_TO_CHOOSE onwards
TrialSpans = bhv.get_spans_from_names(LogDf, "TRIAL_AVAILABLE_STATE", "ITI_STATE")

TrialDfs = []
for i, row in tqdm(TrialSpans.iterrows(),position=0, leave=True):
    TrialDfs.append(bhv.time_slice(LogDf, row['t_on'], row['t_off']))

metrics = (met.get_start, met.get_stop, met.get_correct_side, met.get_interval_category, met.get_outcome, 
            met.get_chosen_side, met.has_reach_left, met.has_reach_right, met.get_in_corr_loop, met.reach_rt_left, 
            met.reach_rt_right, met.has_choice, met.get_interval, met.get_timing_trial, met.get_choice_rt,
            met.get_reached_side)

SessionDf = bhv.parse_trials(TrialDfs, metrics)

# expand outcomes in boolean columns and fixing jackpot rewards
SessionDf.loc[pd.isna(SessionDf['outcome']),['outcome']] = 'jackpot'

outcomes = SessionDf['outcome'].unique()
for outcome in outcomes:
   SessionDf['is_'+outcome] = SessionDf['outcome'] == outcome

# setup general filter
SessionDf['exclude'] = False

#  Plots dir and animal info
animal_meta = pd.read_csv(log_path.parent.parent / 'animal_meta.csv')
nickname = animal_meta[animal_meta['name'] == 'Nickname']['value'].values[0]

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

# %% Do they react (fast) to valve opening?
align_events = ['REWARD_LEFT_VALVE_ON','REWARD_RIGHT_VALVE_ON']
pre,post = 500,2000 # ms

bin_width = 50 # ms

fig, axes = plt.subplots(nrows = 2)

for align_event,axis in zip(align_events,axes):
    bhv_plt_reach.plot_reaches_window_aligned_on_event(LogDf, align_event, pre, post, bin_width,axes = axis)

fig.tight_layout()
plt.savefig(plot_dir / ('hist_of_reaches_aligned_to_valve_opening.png'), dpi=600)

# %% In how many trials do they miss because they have reaches with insufficient duration?
# Doesnt desambiguate which reach it was, just that there was a reach that was shorter than needed

missedDf = SessionDf[SessionDf['outcome'] == 'missed']

# Not considering when they try both sides!
correct_attemptsDf = missedDf[missedDf['correct_side'] == missedDf['reached_side']]

# Considering they try to both
both_side_attemptsDf = missedDf[missedDf['reached_side'] == 'both']

# In how many MISSED trials are there attempts below duration threshold
corr_attempts_perc = round(len(correct_attemptsDf)/len(missedDf)*100,2)

# Or attempts to both sides
both_side_perc = round(len(both_side_attemptsDf)/len(missedDf)*100,2)

print('There were ' + str(corr_attempts_perc)+ '% (putatively correct) trials with reach durations below threshold')
print('And ' + str(both_side_perc) + '% with reaches for both sides')

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
choice_interval = 2000 # ms
bin_width = 100 # ms

bhv_plt_reach.plot_choice_RT_hist(SessionDf, choice_interval, bin_width)
plt.savefig(plot_dir / ('choice_RTs.png'), dpi=600)

# %% Grasp duration distro split by outcome and choice
bin_width = 5 # ms
max_reach_dur = 100 # ms

perc = 25 #th 

bhv_plt_reach.plot_grasp_duration_distro(LogDf, bin_width, max_reach_dur, perc)
plt.savefig(plot_dir / ('hist_grasp_dur_' + str(perc) + '_percentile.png'), dpi=600)

# %% Are they using a sampling strategy? 
fig, axes = plt.subplots(figsize=(3, 4))

missesDf_idx = SessionDf['outcome'] == 'missed'
choiceDf = SessionDf[~missesDf_idx] # drop rows with missed trials

# What is the prob if going left after going right?
has_right_reach_Df = choiceDf[choiceDf['has_reach_right'] == True]
lefts_after_right_df = (has_right_reach_Df['reach_rt_right'] < has_right_reach_Df['reach_rt_left'])
perc_l_after_r = lefts_after_right_df.sum()/len(lefts_after_right_df)*100

# And the inverse here
has_left_reach_Df = choiceDf[choiceDf['has_reach_left'] == True]
rights_after_left_df = (has_left_reach_Df['reach_rt_left'] < has_left_reach_Df['reach_rt_right'])
perc_r_after_l = rights_after_left_df.sum()/len(rights_after_left_df)*100

labels = ['R after L', 'L after R']

# Plotting groups
axes.bar(labels, [perc_r_after_l,perc_l_after_r])

axes.set_title('Prob of going X after Y')
axes.set_ylabel('Prob. (%)')
axes.set_ylim([0,75])
axes.legend(loc='upper left', frameon=False) 

fig.tight_layout()

plt.savefig(plot_dir / ('prob_X_after_Y.png'), dpi=600)


"""
 #       #######    #    ######  #     #    ####### #######    ### #     # ### #######
 #       #         # #   #     # ##    #       #    #     #     #  ##    #  #     #
 #       #        #   #  #     # # #   #       #    #     #     #  # #   #  #     #
 #       #####   #     # ######  #  #  #       #    #     #     #  #  #  #  #     #
 #       #       ####### #   #   #   # #       #    #     #     #  #   # #  #     #
 #       #       #     # #    #  #    ##       #    #     #     #  #    ##  #     #
 ####### ####### #     # #     # #     #       #    #######    ### #     # ###    #

"""

# %% Reaching CDF's for short and long trials (useful before learn to fixate)
bhv_plt_reach.CDF_of_reaches_during_delay(SessionDf,TrialDfs)
fig.suptitle('CDF of first reach split on trial type')
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
ylims = [-3000,2000]
align_event = 'TRIAL_ENTRY_EVENT'
fig, axes = plt.subplots(ncols=2,sharex=True, figsize=(6,3))

linspace = np.linspace(-window, window, 2*window).T

X,Y = bhv_plt_reach.get_LC_slice_aligned_on_event(LoadCellDf, TrialDfs, align_event, window, window) 

# Average traces
X_mean = np.mean(X, axis = 1)
Y_mean = np.mean(Y, axis = 1)

axes[0].plot(linspace, X, alpha=0.25, c = 'tab:orange')
axes[0].plot(linspace, X_mean, c = 'k')
axes[1].plot(linspace, Y, alpha=0.25, c = 'tab:blue')
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
init_tresh = 850
sub_tresh = 0.66*init_tresh # About two thirds the original

fig = plt.figure(figsize=(6,4))

attempts = []
for TrialDf in TrialDfs:

    # Slice when trial is available but not initiated
    Df = bhv.event_slice(TrialDf, "TRIAL_AVAILABLE_EVENT", "TRIAL_ENTRY_EVENT")
    
    if not Df.empty:
        LCDf = bhv.time_slice(LoadCellDf, Df.iloc[0]['t'], Df.iloc[-1]['t'])

        X = LCDf['x'].values
        Y = LCDf['y'].values

        # sub thresh pushing
        X_peaks , _ = scipy.signal.find_peaks(-X, height = sub_tresh, width = 50)
        Y_peaks , _ = scipy.signal.find_peaks(-Y, height = sub_tresh, width = 50)

        attempts.append(max(len(X_peaks),len(Y_peaks)))

    # Instant initiation 
    else:
        attempts.append(0)

# Plotting settings
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
spacing = 0.005
rect_scatter = [left, bottom, width, height]
rect_histy = [left + width + spacing, bottom, 0.2, height]

ax = fig.add_axes(rect_scatter)
ax_histy = fig.add_axes(rect_histy, sharey=ax)

# Formatting
ax.scatter(np.arange(len(TrialDfs)), attempts, s = 2)
ax.set_ylim([0,20])
ax.set_ylabel('No of init attempts')
ax.set_xlabel('Trial No.')

n, bins, _ = ax_histy.hist(attempts, bins=20, range= (0,20), density = True, orientation='horizontal')
ax_histy.tick_params(axis="y", labelleft=False)
ax_histy.set_xlabel('Perc. of trials (%)')
ax_histy.set_xlim([0,0.75])

ax.set_title('Number attempts before trial init')
plt.savefig(plot_dir / ('attempts_before_init.png'), dpi=600)

# %% Trial initiation distro
max_range = 10000 # ms, for actual inits
bin_width = 500 # ms for actual inits
tresh = 850 # for putative init detection
window_width = 250 # ms, for moving average
pre,post = 2000, 2000 # for putative init

align_event = 'TRIAL_AVAILABLE_EVENT'
fig, axes = plt.subplots(ncols=2, figsize=(6,3))

# Putative initiations obtained through LC and thresholding
X,Y = bhv_plt_reach.get_LC_slice_aligned_on_event(LoadCellDf, TrialDfs, align_event, pre, post)

# Threshold needs to be crossed on both
X_tresh_idx = X < -tresh
Y_tresh_idx = Y < -tresh
gate_AND_idx = X_tresh_idx*Y_tresh_idx
gate_AND_idx_sum = np.convolve(gate_AND_idx.sum(axis=1), np.ones(window_width), 'valid') / window_width # moving average

axes[0].plot(gate_AND_idx_sum)
axes[0].set_title('Putative inits aligned \n to trial avail')
axes[0].axvline(pre, linestyle='dotted', color='k', alpha = 0.5)
plt.setp(axes[0], xticks=np.arange(0, post+pre+1, 5000), xticklabels=np.arange(-pre/1000, post/1000 + 0.1, 5))

# Initiations obtained through LogDf
initiation_time = []
for TrialDf in TrialDfs:
    initiation_time.append(met.get_init_rt(TrialDf))

initiation_time = np.array(initiation_time)
axes[1].hist(initiation_time, bins = np.linspace(0,max_range, round(max_range/bin_width)), range = (0,max_range))
axes[1].set_title('Distro of init times')

# Formatting
for ax in axes:
    ax.set_ylabel('No. of ocurrences')
    ax.set_xlabel('Time (s)')

fig.tight_layout()
plt.savefig(plot_dir / ('trial_init_distro.png'), dpi=600)

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
# %% Session overview aligned to 1st and 2nd
pre,post = 500,3000
align_event = 'PRESENT_INTERVAL_STATE'

bhv_plt_reach.plot_session_aligned_to_1st_2nd(LogDf, align_event, pre, post)
plt.savefig(plot_dir / ('session_overview_aligned_on_1st_2nd.png'), dpi=600)

# %% Plot simple session overview with success rate
fig, axes = plt.subplots(figsize=[8,2])

bhv_plt_reach.plot_session_overview(SessionDf, axes = axes)
plt.savefig(plot_dir / ('session_overview.png'), dpi=600)

# %% outcome_split_by_interval_category
bhv_plt_reach.outcome_split_by_interval_category(SessionDf)
plt.savefig(plot_dir / ('outcome_split_by_interval_category.png'), dpi=600)

# %% Psychometric timing trials
TimingDf = SessionDf.groupby('timing_trial').get_group(True)
pre_idx = TimingDf['outcome'] == 'premature'

bhv_plt_reach.plot_psychometric(TimingDf[~pre_idx])
plt.savefig(plot_dir / ('psychometric_timing.png'), dpi=600)

# %% Pseudo-Psychometric with premature timing trials
TimingDf = SessionDf.groupby('timing_trial').get_group(True)
pre_idx = TimingDf['outcome'] == 'premature'

bhv_plt_reach.plot_psychometric(TimingDf[pre_idx])
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
task_name = ['learn_to_choose']
FilteredSessionsDf = pd.concat([SessionsDf.groupby('task').get_group(name) for name in task_name])
log_paths = [Path(path)/'arduino_log.txt' for path in FilteredSessionsDf['path']]

# Obtain the perc of reaches, correct and incorrect trials
perc_reach_right, perc_reach_left, perc_correct, perc_missed, perc_pre = [],[],[],[],[]
date,tpm,session_length = [],[],[] # Trials Per Minute (tpm)

for log_path in tqdm(log_paths):
    
    path = log_path.parent 
    LogDf = bhv.get_LogDf_from_path(log_path)

    # Correct date format
    folder_name = os.path.basename(path)
    complete_date = folder_name.split('_')[0]
    month = calendar.month_abbr[int(complete_date.split('-')[1])]
    day = complete_date.split('-')[2]
    date.append(month+'-'+day)

    TrialSpans = bhv.get_spans_from_names(LogDf, "TRIAL_ENTRY_STATE", "ITI_STATE")

    TrialDfs = []
    for i, row in tqdm(TrialSpans.iterrows(),position=0, leave=True):
        TrialDfs.append(bhv.time_slice(LogDf, row['t_on'], row['t_off']))

    metrics = (met.get_start, met.get_stop, met.get_correct_side, met.get_outcome, met.get_chosen_side, met.has_reach_left, met.has_reach_right)
    SessionDf = bhv.parse_trials(TrialDfs, metrics)

    left_trials_idx = SessionDf['correct_side'] == 'left'
    right_trials_idx = SessionDf['correct_side'] == 'right'

    any_reach = SessionDf.has_reach_left | SessionDf.has_reach_right

    # Two metrics of evolution
    perc_reach_left.append(any_reach[left_trials_idx].sum()/len(SessionDf[left_trials_idx])*100) 
    perc_reach_right.append(any_reach[right_trials_idx].sum()/len(SessionDf[right_trials_idx])*100)
    perc_correct.append((SessionDf.outcome == 'correct').sum()/len(SessionDf)*100)
    perc_missed.append((SessionDf.outcome == 'missed').sum()/len(SessionDf)*100)

    if len(SessionDf[SessionDf.outcome == 'premature']) != 0:
        perc_pre.append(sum(SessionDf.outcome == 'premature')/len(SessionDf)*100)
    else:
        perc_pre.append(0)

    tpm.append(len(SessionDf))
    session_length.append((LogDf['t'].iloc[-1]-LogDf['t'].iloc[0])/(1000*60)) # convert msec. -> sec.-> min.

fig , axes = plt.subplots()

axes.plot(perc_reach_left, color = 'orange', label = 'Reached L (%)', marker='o', markersize=2)
axes.plot(perc_reach_right, color = 'blue', label = 'Reached R (%)', marker='o', markersize=2)
axes.plot(perc_correct, color = 'green', label = 'Correct (%)', marker='o', markersize=2)
axes.plot(perc_missed, color = 'grey', label = 'Missed (%)', marker='o', markersize=2)
axes.plot(perc_pre, color = 'pink', label = 'Premature (%)', marker='o', markersize=2)

axes.set_ylabel('Trial outcome (%)')
axes.set_xlabel('Session date')
axes.legend(loc="center", frameon=False, bbox_to_anchor=(0.5, 1.05), prop={'size': 6}, ncol=5) 

plt.setp(axes, xticks=np.arange(0, len(date), 1), xticklabels=date)
plt.setp(axes, yticks=np.arange(0, 100+1, 10), yticklabels=np.arange(0, 100+1, 10))
plt.xticks(rotation=45)
plt.savefig(across_session_plot_dir / ('overview_across_sessions' + str(task_name)+ '.png'), dpi=600)

# %% Trials per minute
fig , axes = plt.subplots()
axes.plot(np.array(tpm)/np.array(session_length))

axes.set_ylim([0,5]) # A trial every 15s is the theoretical limit at chance level given our task settings
plt.title('No. TPM across sessions')
plt.setp(axes, xticks=np.arange(0, len(date), 1), xticklabels=date)
plt.xticks(rotation=45)

plt.savefig(across_session_plot_dir / ('tpm_across_sessions.png'), dpi=600)

# %% Reaches during delay across sessions
SessionsDf = utils.get_sessions(animal_fd_path)
Df = pd.read_csv(animal_fd_path / 'animal_meta.csv')

# Filter sessions to the ones of the task we want to see
task_name = ['learn_to_choose']
FilteredSessionsDf = pd.concat([SessionsDf.groupby('task').get_group(name) for name in task_name])
log_paths = [Path(path)/'arduino_log.txt' for path in FilteredSessionsDf['path']]

fig, axes = plt.subplots(ncols=2, figsize=[6, 3], sharey=True, sharex=True)
colors = sns.color_palette(palette='turbo',n_colors=len(log_paths))

for j,log_path in enumerate(log_paths):
    
    path = log_path.parent 
    LogDf = bhv.get_LogDf_from_path(log_path)

    # ADD SINGLE GO_CUE_EVENT
    LogDf = bhv_plt_reach.add_go_cue_LogDf(LogDf)

    folder_name = os.path.basename(path)

    TrialSpans = bhv.get_spans_from_names(LogDf, "TRIAL_ENTRY_STATE", "ITI_STATE")

    TrialDfs = []
    for i, row in tqdm(TrialSpans.iterrows(),position=0, leave=True):
        TrialDfs.append(bhv.time_slice(LogDf, row['t_on'], row['t_off']))

    metrics = (met.get_start, met.get_stop, met.get_correct_side, met.get_outcome, met.get_interval_category, met.get_chosen_side, met.has_reach_left, met.has_reach_right)
    SessionDf = bhv.parse_trials(TrialDfs, metrics)

    bhv_plt_reach.CDF_of_reaches_during_delay(SessionDf,TrialDfs, axes = axes, color=colors[j], alpha=0.75, label='day '+str(j))
    fig.suptitle('CDF of first reach split on trial type')

axes[0].set_ylabel('Fraction of trials')
axes[0].legend(frameon=False, fontsize='x-small')
fig.tight_layout()

plt.savefig(across_session_plot_dir / ('CDF_first_reach_across_sessions.png'), dpi=600)


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
for color_no, path in enumerate(paths):

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
