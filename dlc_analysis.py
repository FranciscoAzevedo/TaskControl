# %% imports 
%matplotlib qt5
%matplotlib qt5
%load_ext autoreload
%autoreload 2

import matplotlib.pyplot as plt
from matplotlib import cm
import scipy as sp
import numpy as np
import pandas as pd
import cv2
import utils
import calendar
import os

from tqdm import tqdm
from pathlib import Path

# Custom
import behavior_analysis_utils as bhv
from dlc_analysis_utils import *
import behav_plotters_reach as bhv_plt_reach

# Settings
# Plotting Defaults
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.major.size"] = 1.5
plt.rcParams["ytick.major.size"] = 1.5
plt.rcParams['figure.dpi'] = 166

"""
 #       #######    #    ######  ### #     #  #####
 #       #     #   # #   #     #  #  ##    # #     #
 #       #     #  #   #  #     #  #  # #   # #
 #       #     # #     # #     #  #  #  #  # #  ####
 #       #     # ####### #     #  #  #   # # #     #
 #       #     # #     # #     #  #  #    ## #     #
 ####### ####### #     # ######  ### #     #  #####

"""

# %% read all four data sources (Video, DLC markers, Loadcells and Logs)

# DeepLabCut data
h5_path = utils.get_file_dialog()
#DlcDf = read_dlc_h5(h5_path)
#bodyparts = sp.unique([j[0] for j in DlcDf.columns[1:]]) # all body parts

# Video
path = h5_path.parent
video_path = path / "bonsai_video.avi"
Vid = read_video(str(video_path))

# Logs
log_path = path / 'arduino_log.txt'
LogDf = bhv.get_LogDf_from_path(log_path)

# LoadCell data
LoadCellDf, t_harp = bhv.parse_bonsai_LoadCellData(path / "bonsai_LoadCellData.csv",trig_len=100, ttol=4)

# Moving average mean subtraction
samples = 1000 # ms
LoadCellDf['x'] = LoadCellDf['x'] - LoadCellDf['x'].rolling(samples).mean()
LoadCellDf['y'] = LoadCellDf['y'] - LoadCellDf['y'].rolling(samples).mean()

# %% Synching 
video_sync_path = video_path.parent / 'bonsai_frame_stamps.csv'
m, b, m2, b2 = sync_arduino_w_dlc(log_path, video_sync_path)

# writing arduino times of frames to the Dlc data
#DlcDf['t'] = frame2time(DlcDf.index,m,b,m2,b2)

# Synching arduino 
arduino_sync = bhv.get_arduino_sync(log_path, sync_event_name="TRIAL_ENTRY_EVENT")
t_harp = t_harp['t'].values
t_arduino = arduino_sync['t'].values

if t_harp.shape != t_arduino.shape:
    t_arduino, t_harp = bhv.cut_timestamps(t_arduino, t_harp, verbose = True)

m3, b3 = bhv.sync_clocks(t_harp, t_arduino, log_path = log_path)
LogDf = pd.read_csv(path / "LogDf.csv") # re-load the LogDf (to make sure we keep the original arduino clock)

# %% Create SessionDf
TrialSpans = bhv.get_spans_from_names(LogDf, "TRIAL_AVAILABLE_STATE", "ITI_STATE")

TrialDfs = []
for i, row in tqdm(TrialSpans.iterrows(),position=0, leave=True):
    TrialDfs.append(bhv.time_slice(LogDf, row['t_on'], row['t_off']))

metrics = (bhv.get_start, bhv.get_stop, get_trial_side, get_trial_type, get_outcome, get_choice, has_reach_left, has_reach_right, \
           bhv.get_in_corr_loop, choice_rt_left, choice_rt_right, bhv.has_choice, bhv.get_interval, get_timing_trial)

SessionDf = bhv.parse_trials(TrialDfs, metrics)

# %% Plots dir and animal info
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
  #####  #######  #####   #####  ### ####### #     #             #     # ### ######  ####### #######
 #     # #       #     # #     #  #  #     # ##    #             #     #  #  #     # #       #     #
 #       #       #       #        #  #     # # #   #             #     #  #  #     # #       #     #
  #####  #####    #####   #####   #  #     # #  #  #    #####    #     #  #  #     # #####   #     #
       # #             #       #  #  #     # #   # #              #   #   #  #     # #       #     #
 #     # #       #     # #     #  #  #     # #    ##               # #    #  #     # #       #     #
  #####  #######  #####   #####  ### ####### #     #                #    ### ######  ####### #######

"""

# %% plot a single frame with DLC markers and Skeleton
fig, axes = plt.subplots()
i = 8000 # frame index
Frame = get_frame(Vid, i)
axes = plot_frame(Frame, axes=axes)
axes = plot_bodyparts(bodyparts, DlcDf, i, axes=axes)
axes, lines = plot_Skeleton(Skeleton, DlcDf, i , axes=axes)

# %% plot a heatmap of movement for both paws on a 2D background
fig, axes = plt.subplots()

i = 4000 # frame index
Frame = get_frame(Vid, i)
axes = plot_frame(Frame, axes=axes)
axes = plot_trajectories(DlcDf, paws, axes=axes,lw=0.025)
axes.axis('off')
axes.set_title('Whole session heatmap of paw placement')

plt.savefig(plot_dir / ('heatmap_both_paws.png'), dpi=600)

# %% Plot reaches to one side with one paw

# Settings
right_spout = [230,400] # spout right
#left_spout = [380, 405] # spout left
avg_mvmt_time = 250
p = 0.99

# Background image
fig, axes = plt.subplots()
i = 8000 # frame index
Frame = get_frame(Vid, i)
axes = plot_frame(Frame, axes=axes)

# Detection rectangle
w = 75 # box size
rect = box2rect(right_spout, w)
R = Rectangle(*rect2cart(rect),lw=1,facecolor='none',edgecolor='r')
axes.add_patch(R)

# Obtain all reaches within rectangle, convert from frame to time
bp = 'PR'
SpansDf = in_box_span(DlcDf, bp, rect, min_dur=5)
SpansDf = pd.DataFrame(frame2time(SpansDf.values,m,b,m2,b2),columns=SpansDf.columns)

# Plot all reaches to given side 
df = DlcDf[bp]
for i, row in tqdm(SpansDf.iterrows()):
    t_on = row['t_on']
    df = bhv.time_slice(DlcDf,t_on-avg_mvmt_time,t_on)[bp]

    ix = df.likelihood > p
    df = df.loc[ix]
    axes.plot(df.x,df.y,lw=0.2, alpha=0.85, c = 'tab:blue')

axes.set_title('Reach trajectories for right_spout with ' + str(bp))
plt.savefig(plot_dir / ('reaches_for_right_spout_with_' + str(bp) + '.png'), dpi=600)

# %% distance / speed over time
fig, axes = plt.subplots(nrows=2,sharex=True)

bps = ['PR','PL']
right_spout = [230,400]
left_spout = [380, 405]

line_kwargs = dict(lw=1,alpha=0.8)
for i, bp in enumerate(bps):
    d_to_right = calc_dist_bp_point(DlcDf, bp, right_spout, filter=True)
    d_to_left = calc_dist_bp_point(DlcDf, bp, left_spout, filter=True)
    axes[i].plot(d_to_left, label='to left', **line_kwargs)
    axes[i].plot(d_to_right, label='to right', **line_kwargs)
    axes[i].set_ylabel(bp)
    axes[i].set_ylim(0)

axes[0].legend()

# %% Distance aligned to align_event split by side
pre,post = 1000,3000
align_event = 'PRESENT_CUE_STATE'
time_interval = 1000 # ms (for the time axis in the plot)

fig, axes = plt.subplots(ncols=2,sharex=True)

TrialDfs_left = filter_trials_by(SessionDf, TrialDfs, ('trial_side', 'left'))
TrialDfs_right = filter_trials_by(SessionDf, TrialDfs, ('trial_side', 'right'))

# General function to be applied 
func = calc_dist_bp_point

d_to_left = get_dist_aligned_on_event(DlcDf, TrialDfs_left, align_event, pre, post, func, 'PR', left_spout)
d_to_right = get_dist_aligned_on_event(DlcDf, TrialDfs_right, align_event, pre, post, func, 'PR', right_spout)

heat1 = axes[0].matshow(d_to_left, vmin=0, vmax=200, cmap='viridis_r', extent=[-pre,post,0,d_to_left.shape[0]])
heat2 = axes[1].matshow(d_to_right, vmin=0, vmax=150, cmap='viridis_r', extent=[-pre,post,0,d_to_right.shape[0]])

cbar1 = plt.colorbar(heat1, ax=axes[0], orientation='horizontal', aspect = 30)
cbar2 = plt.colorbar(heat2, ax=axes[1], orientation='horizontal', aspect = 30)

cbar1.ax.set_xlabel('Euclid. Distance (a.u.)')
cbar2.ax.set_xlabel('Euclid. Distance (a.u.)')

axes[0].set_title('Left Trials')
axes[0].set_ylabel('Trials')
axes[1].set_title('Right Trials')

for ax in axes.flatten():
    ax.set_xlabel('Time (s)')
    ax.set_aspect('auto')

for ax in axes:
    ax.axvline(x=0, ymin=0, ymax=1, color = 'red', alpha = 0.5)
    plt.setp(ax, xticks=np.arange(-pre, post+1, time_interval), xticklabels=np.arange(-pre/1000, post/1000+0.1, time_interval/1000))
    ax.xaxis.set_ticks_position('bottom')

plt.savefig(plot_dir / ('paw_distance_aligned_' + str(align_event) + '.png'), dpi=600)

# %% Are they priming actions by having a specific posture for each trial type?
align_event = 'PRESENT_CUE_STATE'

fig, axes = plt.subplots(ncols=2, figsize=(5, 3))

TrialDfs_left = filter_trials_by(SessionDf, TrialDfs, ('choice', 'left'))
TrialDfs_right = filter_trials_by(SessionDf, TrialDfs, ('choice', 'right'))

pl, pr = [],[]

# For every trial
for TrialDf in TrialDfs_left:

    # get timepoint of cue presentation
    log_t_align = bhv.get_events_from_name(TrialDf, align_event)['t']

    # slice the DlcDf at that point (or seach the nearest by subtracting
    # and finding the lowest value which corresponds to the shift between LogDf and DlcDf)
    Dlc_idx = DlcDf['t'].sub(int(log_t_align)).abs().idxmin()
    DlcDf_slice = DlcDf.iloc[Dlc_idx]

    # Plot the locations of the L/R paws separately for L/R trials just before cue presentation
    pl.append([DlcDf_slice['PL']['x'],DlcDf_slice['PL']['y']])
    pr.append([DlcDf_slice['PR']['x'],DlcDf_slice['PR']['y']])

pl = np.array(pl)
pr = np.array(pr)

# Plot the locations of the L/R paws separately for L/R trials just before cue presentation
axes[0].scatter(pl[:,0], pl[:,1], s = 1, alpha = 0.75, c = 'tab:blue', label = 'Left Paw')
axes[0].scatter(pr[:,0], pr[:,1], s = 1, alpha = 0.75, c = 'tab:orange', label = 'Right Paw')

pl, pr = [],[]
# For every trial
for TrialDf in TrialDfs_right:

    # get timepoint of cue presentation
    log_t_align = bhv.get_events_from_name(TrialDf, align_event)['t']

    # slice the DlcDf at that point (or seach the nearest by subtracting
    # and finding the lowest value which corresponds to the shift between LogDf and DlcDf)
    Dlc_idx = DlcDf['t'].sub(int(log_t_align)).abs().idxmin()
    DlcDf_slice = DlcDf.iloc[Dlc_idx]

    # Plot the locations of the L/R paws separately for L/R trials just before cue presentation
    pl.append([DlcDf_slice['PL']['x'],DlcDf_slice['PL']['y']])
    pr.append([DlcDf_slice['PR']['x'],DlcDf_slice['PR']['y']])

pl = np.array(pl)
pr = np.array(pr)

# Plot the locations of the L/R paws separately for L/R trials just before cue presentation
axes[1].scatter(pl[:,0], pl[:,1], s = 1, alpha = 0.75, c = 'tab:blue', label = 'Left Paw')
axes[1].scatter(pr[:,0], pr[:,1], s = 1, alpha = 0.75, c = 'tab:orange', label = 'Right Paw')

# Plot a single fram in the background for comparison
i = 4000 # frame index
Frame = get_frame(Vid, i)
axes[0] = plot_frame(Frame, axes=axes[0])
axes[1] = plot_frame(Frame, axes=axes[1])

# Formatting
axes[0].set_title('Left choice')
axes[1].set_title('Right choice')

for ax in axes:
    ax.legend(loc="center", bbox_to_anchor=(0.5, -0.2), prop={'size': 8}, ncol=2, frameon= False)
    ax.axis('off')

fig.suptitle('Paw placement aligned to ' + align_event)

plt.savefig(plot_dir / ('paw_placement_aligned_to_' + align_event + '.png'), dpi=600)



"""
  #####  #######  #####   #####  ### ####### #     #             #       #######  #####
 #     # #       #     # #     #  #  #     # ##    #             #       #     # #     #
 #       #       #       #        #  #     # # #   #             #       #     # #
  #####  #####    #####   #####   #  #     # #  #  #    #####    #       #     # #  ####
       # #             #       #  #  #     # #   # #             #       #     # #     #
 #     # #       #     # #     #  #  #     # #    ##             #       #     # #     #
  #####  #######  #####   #####  ### ####### #     #             ####### #######  #####

"""

# %% Success rate
bhv_plt_reach.plot_success_rate(LogDf, SessionDf, 10, axes=None)
plt.savefig(plot_dir / ('success_rate.png'), dpi=600)

# %% Session overview
pre,post = 500,3000
align_event = 'PRESENT_INTERVAL_STATE'

bhv_plt_reach.plot_session_overview(LogDf, align_event, pre, post)
plt.savefig(plot_dir / ('session_overview.png'), dpi=600)

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

# %% Psychometric for cued trials
bhv_plt_reach.plot_psychometric(SessionDf.groupby('timing_trial').get_group(False))
plt.savefig(plot_dir / ('psychometric_cued.png'), dpi=600)

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

# %% 1st reach choice RT
choice_interval = 1000 # ms
bin_width = 50 # ms

bhv_plt_reach.plot_choice_RT_hist(SessionDf, choice_interval, bin_width)
plt.savefig(plot_dir / ('choice_RTs.png'), dpi=600)

# %% Reaches in delay period (useful before learn to fixate)
event_1 = 'PRESENT_INTERVAL_STATE'
event_2 = 'GO_CUE_EVENT'
bin_width = 50 # ms

bhv_plt_reach.plot_reaches_between_events(SessionDf, LogDf, TrialDfs, event_1, event_2, bin_width)
plt.savefig(plot_dir / ('hist_of_reaches_between_' + str(event_1) + '_and_' + str(event_2) + '.png'), dpi=600)

# %% Reaching CDF's for short and long trials (useful before learn to fixate)
trial_types = ['short','long']
fig, axes = plt.subplots(figsize=(4, 4))

for i, trial_type in enumerate(trial_types):

    TrialDfs_type =  filter_trials_by(SessionDf, TrialDfs, ('trial_type', trial_type))

    rts = []
    for TrialDf_type in TrialDfs_type:
        rts.append(reach_rt_delay_period(TrialDf_type))

    rts = np.array(rts)
    count, bins_count = np.histogram(rts[~np.isnan(rts)])
    pdf = count / len(rts) # sum of RT's since I want to include the Nans - normalize across no. trials
    cdf = np.cumsum(pdf)

    axes.plot(bins_count[1:], cdf, label=trial_type)

# Formatting
axes.set_xlabel('Time since 1st cue (ms)')
axes.set_title('CDF of first reach split on trial type')
axes.set_ylim([0,1])
axes.legend(loc='upper left', frameon=False)
plt.savefig(plot_dir / ('CDF_of_reach_during_delay.png'), dpi=600)

# %% Reaches in a window around align_event 
pre, post = 500,3000 # ms
align_event = 'PRESENT_INTERVAL_STATE'
filter_pair = ('trial_side', 'right')

bhv_plt_reach.plot_reaches_window_aligned_on_event(SessionDf, TrialDfs, align_event, filter_pair, pre, post)
plt.savefig(plot_dir / ('reach_distro_aligned_' + str(align_event) + '_split_by_' + str(filter_pair) + '_trials.png'), dpi=600)

# %% Reach duration distro split by outcome and trial side
choices = ['left', 'right']
outcomes = ['correct', 'incorrect']

bin_width = 100 # ms
max_reach_dur = 2000 # ms

fig, axes = plt.subplots(nrows=len(outcomes), ncols=len(choices), figsize=[4, 4], sharex=True, sharey=True)

no_bins = round(max_reach_dur/bin_width)

kwargs = dict(bins = no_bins, range = (0, max_reach_dur), alpha=0.5, edgecolor='none')

# for each condition pair
for i, choice in enumerate(choices):
    for j, outcome in enumerate(outcomes):
        
        # try to filter trials
        try:
            SDf = SessionDf.groupby(['choice', 'outcome']).get_group((choice, outcome))
        except:
            continue
        
        ax = axes[j, i]
    
    # get spans Df for reaches

    # hist plot of duration of reaches
    

# %% How hard do they push to initiate trials?
window = 300 # ms
ylims = [-3000,2000]
align_event = 'TRIAL_ENTRY_EVENT'
fig, axes = plt.subplots(ncols=2,sharex=True, figsize=(6,3))

linspace = np.linspace(-window, window, 2*window).T

X,Y = get_LC_window_aligned_on_event(LoadCellDf, TrialDfs, align_event, window, window) 

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

# %% Trial initiation distro
max_range = 5000 # ms, for actual inits
bin_width = 500 # ms for actual inits
tresh = 500 # for putative init detection
window_width = 250 # ms, for moving average
pre,post = 2000, 2000 # for putative init

align_event = 'TRIAL_AVAILABLE_EVENT'
fig, axes = plt.subplots(ncols=2, figsize=(6,3))

# Putative initiations obtained through LC and thresholding
X,Y = get_LC_window_aligned_on_event(LoadCellDf, TrialDfs, align_event, pre, post)

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
    t_trial_available = TrialDf[TrialDf['name'] == 'TRIAL_AVAILABLE_EVENT']['t'].values
    t_trial_entry = TrialDf[TrialDf['name'] == 'TRIAL_ENTRY_EVENT']['t'].values

    initiation_time.append(t_trial_entry-t_trial_available)

initiation_time = np.array(initiation_time)
axes[1].hist(initiation_time, bins = np.linspace(0,max_range, round(max_range/bin_width)), range = (0,max_range))
axes[1].set_title('Distro of init times')

# Formatting
for ax in axes:
    ax.set_ylabel('No. of ocurrences')
    ax.set_xlabel('Time (s)')

fig.tight_layout()
plt.savefig(plot_dir / ('trial_init_distro.png'), dpi=600)

# %% Are they using a sampling strategy? 
fig, axes = plt.subplots(figsize=(3, 4))

missesDf_idx = SessionDf['outcome'] == 'missed'
choiceDf = SessionDf[~missesDf_idx] # drop rows with missed trials

sides = ['left', 'right']

perc_r_after_l, perc_l_after_r = [],[]
for i, side in enumerate(sides):

    trial_sideDf = choiceDf[choiceDf['trial_side'] == side]

    has_left_reach_Df = trial_sideDf[trial_sideDf['has_reach_left'] == True]
    has_right_reach_Df = trial_sideDf[trial_sideDf['has_reach_right'] == True]

    # What is the prob of going right after going left?
    rights_after_left_df = (has_left_reach_Df['choice_rt_left'] < has_left_reach_Df['choice_rt_right'])
    perc_r_after_l.append(sum(rights_after_left_df)/len(rights_after_left_df)*100)

    # What is the prob if going left after going right?
    lefts_after_right_df = (has_right_reach_Df['choice_rt_right'] < has_right_reach_Df['choice_rt_left'])
    perc_l_after_r.append(sum(lefts_after_right_df)/len(lefts_after_right_df)*100)

labels = ['left trials', 'right trials']
x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

# Plotting groups
rects1 = axes.bar(x - width/2, perc_r_after_l, width, label='R after L')
rects2 = axes.bar(x + width/2, perc_l_after_r, width, label='L after R')

axes.set_title('Prob of going X after Y')
axes.set_ylabel('Prob. (%)')
axes.set_ylim([0,75])
axes.set_xticks(x)
axes.set_xticklabels(labels)
axes.legend(loc='upper left', frameon=False) 

fig.tight_layout()

plt.savefig(plot_dir / ('prob_X_after_Y.png'), dpi=600)

"""
    #     #####  ######  #######  #####   #####      #####  #######  #####   #####  ### ####### #     #  #####
   # #   #     # #     # #     # #     # #     #    #     # #       #     # #     #  #  #     # ##    # #     #
  #   #  #       #     # #     # #       #          #       #       #       #        #  #     # # #   # #
 #     # #       ######  #     #  #####   #####      #####  #####    #####   #####   #  #     # #  #  #  #####
 ####### #       #   #   #     #       #       #          # #             #       #  #  #     # #   # #       #
 #     # #     # #    #  #     # #     # #     #    #     # #       #     # #     #  #  #     # #    ## #     #
 #     #  #####  #     # #######  #####   #####      #####  #######  #####   #####  ### ####### #     #  #####

"""

# %% Loading

# Obtain log_paths and plot dirs
animal_fd_path = utils.get_folder_dialog()
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
task_name = ['learn_to_time','learn_to_time_v2','learn_to_fixate_v1']
FilteredSessionsDf = pd.concat([SessionsDf.groupby('task').get_group(name) for name in task_name])
log_paths = [Path(path)/'arduino_log.txt' for path in FilteredSessionsDf['path']]

# Obtain the perc of reaches, correct and incorrect trials
perc_reach_right, perc_reach_left, perc_correct, perc_missed = [],[],[],[]
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

    metrics = (bhv.get_start, bhv.get_stop, get_trial_side, get_outcome, get_choice, has_reach_left, has_reach_right)
    SessionDf = bhv.parse_trials(TrialDfs, metrics)

    left_trials_idx = SessionDf['trial_side'] == 'left'
    right_trials_idx = SessionDf['trial_side'] == 'right'

    any_reach = SessionDf.has_reach_left | SessionDf.has_reach_right

    # Two metrics of evolution
    perc_reach_left.append(sum(any_reach[left_trials_idx])/len(SessionDf[left_trials_idx])*100) 
    perc_reach_right.append(sum(any_reach[right_trials_idx])/len(SessionDf[right_trials_idx])*100)
    perc_correct.append(sum(SessionDf.outcome == 'correct')/len(SessionDf)*100)
    perc_missed.append(sum(SessionDf.outcome == 'missed')/len(SessionDf)*100)

    tpm.append(len(SessionDf))
    session_length.append((LogDf['t'].iloc[-1]-LogDf['t'].iloc[0])/(1000*60)) # convert msec. -> sec.-> min.

fig , axes = plt.subplots()

axes.plot(perc_reach_left, color = 'orange', label = 'Reached Right (%)')
axes.plot(perc_reach_right, color = 'blue', label = 'Reached Left (%)')
axes.plot(perc_correct, color = 'green', label = 'Correct (%)')
axes.plot(perc_missed, color = 'grey', label = 'Missed (%)')

axes.set_ylabel('Trial outcome (%)')
axes.set_xlabel('Session date')
axes.legend(loc='upper left', frameon=False) 

plt.setp(axes, xticks=np.arange(0, len(date), 1), xticklabels=date)
plt.setp(axes, yticks=np.arange(0, 100+1, 10), yticklabels=np.arange(0, 100+1, 10))
plt.xticks(rotation=45)
plt.savefig(across_session_plot_dir / ('overview_across_sessions.png'), dpi=600)

# %% Trials per minute
fig , axes = plt.subplots()
axes.plot(np.array(tpm)/np.array(session_length))

axes.set_ylim([0,4]) # A trial every 15s is the theoretical limit at chance level given our task settings
plt.title('No. TPM across sessions')
plt.setp(axes, xticks=np.arange(0, len(date), 1), xticklabels=date)
plt.xticks(rotation=45)

plt.savefig(across_session_plot_dir / ('tpm_across_sessions.png'), dpi=600)

# %%
dists = calc_dist_bp_point(DlcDf, bp, coords, p=0.1, filter=True)

for i in range(len(inds)):
    shape = inds[i].shape[0]
    D[:shape,i] = dists[inds[i]]

# %% full plot
fig, axes = plt.subplots(nrows=2,ncols=2,sharex=True,sharey=True)

# events = ['REWARD_LEFT_AVAILABLE_EVENT','REWARD_RIGHT_AVAILABLE_EVENT']
events = ['REWARD_LEFT_VALVE_ON','REWARD_RIGHT_VALVE_ON']
prev_events = ['REWARD_LEFT_AVAILABLE_EVENT','REWARD_RIGHT_AVAILABLE_EVENT']
sides = ['left','right']
coords = [[385, 375],[201,381]] # left, right
pre, post = -7500, 7500
bp = 'PR'

for i,event in enumerate(events):
    for j, point in enumerate(coords):

        # get timestamps
        Event = bhv.get_events_from_name(LogDf, event)

        Prev_Event = bhv.get_events_from_name(LogDf, prev_events[i])

        # to indices
        inds = []
        t_prev = []
        for t in Event.t:
            df = bhv.time_slice(DlcDf, t+pre, t+post)
            inds.append(df.index)

            t_prev.append(bhv.time_slice(Prev_Event,t+pre, t+post)['t']-t)

        # prealloc empty
        D = sp.zeros((np.max([ix.shape[0] for ix in inds]),len(inds)))
        D[:] = sp.nan

        # euclid dist
        dists = calc_dist_bp_point(DlcDf, bp, point, p=0.1, filter=True)

        for k in range(len(inds)):
            shape = inds[k].shape[0]
            D[:shape,k] = dists[inds[k]]

        axes[j,i].matshow(D.T,cmap='viridis_r',vmin=0,vmax=100, origin='bottom', extent=(pre,post,0,D.shape[1]))

        for k in range(len(t_prev)):
            try:
                for q in t_prev[k].values:
                    axes[j,i].plot([q,q],[k,k+1],color='r', alpha=0.5,lw=1)
                    # axes[j,i].plot(t_prev[k],[k]*t_prev[k].shape[0],)
            except:
                pass

for ax in axes.flatten():
    ax.set_aspect('auto')
    ax.axvline(0,alpha=0.5,color='k',linestyle=':')

for i,ax in enumerate(axes[:,0]):
    ax.set_ylabel("to spout %s" % sides[i])

for i,ax in enumerate(axes[0,:]):
    ax.set_title(events[i])

fig.suptitle(bp)
fig.tight_layout()

# %%
Event = bhv.get_events_from_name(LogDf, 'REWARD_LEFT_VALVE_ON')
prev_Event =  bhv.get_events_from_name(LogDf, 'REWARD_LEFT_AVAILABLE_EVENT')
min_dur = 5000

# %%
t = bhv.get_events_from_name(LogDf, "REWARD_LEFT_VALVE_ON").iloc[20].values[0]
# """
 
#     ###    ##    ## #### ##     ##    ###    ######## ####  #######  ##    ## 
#    ## ##   ###   ##  ##  ###   ###   ## ##      ##     ##  ##     ## ###   ## 
#   ##   ##  ####  ##  ##  #### ####  ##   ##     ##     ##  ##     ## ####  ## 
#  ##     ## ## ## ##  ##  ## ### ## ##     ##    ##     ##  ##     ## ## ## ## 
#  ######### ##  ####  ##  ##     ## #########    ##     ##  ##     ## ##  #### 
#  ##     ## ##   ###  ##  ##     ## ##     ##    ##     ##  ##     ## ##   ### 
#  ##     ## ##    ## #### ##     ## ##     ##    ##    ####  #######  ##    ## 
 
# """
# %% play frames
from matplotlib.animation import FuncAnimation
# ix = list(range(30100,30200))
ix = list(range(572,579))

fig, ax = plt.subplots()
ax.set_aspect('equal')
frame = get_frame(Vid, ix[0])
im = ax.imshow(frame, cmap='gray')
# ax, lines = plot_Skeleton(Skeleton, DlcDf, ix[0], axes=ax)

def update(i):
    Frame = get_frame(Vid,i)
    im.set_data(Frame)
    # ax, lines_new = plot_Skeleton(Skeleton, DlcDf, i, axes=ax)
    # for j, line in enumerate(lines):
    #     x = [DlcDf[Skeleton[j][0]].loc[i].x,DlcDf[Skeleton[j][1]].loc[i].x]
    #     y = [DlcDf[Skeleton[j][0]].loc[i].y,DlcDf[Skeleton[j][1]].loc[i].y]
    #     line.set_data(x,y)

    # return im, lines,
    return im,

ani = FuncAnimation(fig, update, frames=ix, blit=True, interval=2)
plt.show()

# # %%
# ani.save('test.avi',fps=30,dpi=300)

# %%