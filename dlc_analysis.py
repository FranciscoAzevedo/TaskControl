# %% imports 
%matplotlib qt5
%matplotlib qt5
%load_ext autoreload
%autoreload 2

# System libs
import os
import cv2
from tqdm import tqdm
from pathlib import Path

# Plotting and math libs
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy as sp
import numpy as np
import pandas as pd
import seaborn as sns

# Custom
from Utils import behavior_analysis_utils as bhv
from Utils import dlc_analysis_utils as dlc
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
fd_path = utils.get_folder_dialog(initial_dir="/media/storage/shared-paton/georg/Animals_reaching/")

# DeepLabCut data and settings
try:
    h5_path = fd_path / [fname for fname in os.listdir(fd_path) if fname.endswith('filtered.h55')][0]
except IndexError:
    h5_path = fd_path / [fname for fname in os.listdir(fd_path) if fname.endswith('.h5')][0]

DlcDf = dlc.read_dlc_h5(h5_path)
bodyparts = np.unique([j[0] for j in DlcDf.columns[1:]]) # all body parts
paws = ['PL','PR']
spouts = ['SL','SR']

# Video data
video_path = fd_path / "bonsai_video.avi"
Vid = dlc.read_video(str(video_path))

# Arduino data
log_path = fd_path / 'arduino_log.txt'
LogDf = bhv.get_LogDf_from_path(log_path)

# LoadCell data
#LoadCellDf = bhv.parse_bonsai_LoadCellData(fd_path / 'bonsai_LoadCellData.csv')

## Synching 

from Utils import sync

# Parse sync events/triggers
cam_sync_event = sync.parse_cam_sync(fd_path / 'bonsai_frame_stamps.csv')
#lc_sync_event = sync.parse_harp_sync(fd_path / 'bonsai_harp_sync.csv')
arduino_sync_event = sync.get_arduino_sync(fd_path / 'arduino_log.txt')

# Get the values out of them
Sync = sync.Syncer()
Sync.data['arduino'] = arduino_sync_event['t'].values
#Sync.data['loadcell'] = lc_sync_event['t'].values
Sync.data['dlc'] = cam_sync_event.index.values # the frames are the DLC
Sync.data['cam'] = cam_sync_event['t'].values # used for what?

# Sync them all to master clock (arduino [FSM?] at ~1Khz) and convert
#LoadCellDf['t_loadcell'] = LoadCellDf['t'] # keeps the original

#Sync.sync('loadcell','arduino')
Sync.sync('dlc','arduino')

#LoadCellDf['t'] = Sync.convert(LoadCellDf['t'].values, 'loadcell', 'arduino')
DlcDf['t'] = Sync.convert(DlcDf.index.values, 'dlc', 'arduino')

## SessionDf and Go cue

# Add single GO_CUE_EVENT
LogDf = bhv.add_go_cue_LogDf(LogDf)

# Moving average mean subtraction (need to do after synching)
samples = 1000 # ms
#LoadCellDf['x'] = LoadCellDf['x'] - LoadCellDf['x'].rolling(samples).mean()
#LoadCellDf['y'] = LoadCellDf['y'] - LoadCellDf['y'].rolling(samples).mean()

#  Create SessionDf 
session_metrics = ( met.get_start, met.get_stop, met.get_correct_side, met.get_interval_category, met.get_outcome, 
            met.get_chosen_side, met.has_reach_left, met.has_reach_right, met.get_in_corr_loop,  
            met.reach_rt_left, met.reach_rt_right, met.has_choice, met.get_interval, met.get_timing_trial,
            met.get_choice_rt, met.get_reached_side, met.get_bias, met.get_init_rt) 

SessionDf, TrialDfs = utils.get_SessionDf(LogDf, session_metrics, "TRIAL_ENTRY_EVENT", "ITI_STATE")

#  Plots dir and animal info
animal_meta = pd.read_csv(log_path.parent.parent / 'animal_meta.csv')
nickname = animal_meta[animal_meta['name'] == 'Nickname']['value'].values[0]
session_date = log_path.parent.stem.split('_')[0]

plot_dir = log_path.parent / 'plots'
os.makedirs(plot_dir, exist_ok=True)

# %%
"""
..######.....###....##....##.####.########.##....##
.##....##...##.##...###...##..##.....##.....##..##.
.##........##...##..####..##..##.....##......####..
..######..##.....##.##.##.##..##.....##.......##...
.......##.#########.##..####..##.....##.......##...
.##....##.##.....##.##...###..##.....##.......##...
..######..##.....##.##....##.####....##.......##...
"""

# Define the point based on mean of highest likelihood
high_likeli_right = DlcDf['SL'][DlcDf['SL']['likelihood'] > 0.99]
SL_coords = [np.mean(high_likeli_right['x'].values), np.mean(high_likeli_right['y'].values)]

high_likeli_right = DlcDf['SR'][DlcDf['SR']['likelihood'] > 0.99]
SR_coords = [np.mean(high_likeli_right['x'].values), np.mean(high_likeli_right['y'].values)]

# %% 
fig, axes = plt.subplots(ncols=2, figsize=(6,4))

for i, paw in enumerate(paws): # loop rows
        axes[i].hist(DlcDf[paw]['likelihood'].values, bins = 10)
        axes[i].set_title(paw)

for ax in axes:
    ax.set_xlabel('likelihood')

axes[0].set_ylabel('Count')
fig.suptitle('How good are the labels?')
fig.tight_layout()

# %%
"""
.########..##........#######..########.########.####.##....##..######..
.##.....##.##.......##.....##....##.......##.....##..###...##.##....##.
.##.....##.##.......##.....##....##.......##.....##..####..##.##.......
.########..##.......##.....##....##.......##.....##..##.##.##.##...####
.##........##.......##.....##....##.......##.....##..##..####.##....##.
.##........##.......##.....##....##.......##.....##..##...###.##....##.
.##........########..#######.....##.......##....####.##....##..######..
"""
# %% plot a single frame with DLC markers and Skeleton
fig, axes = plt.subplots()
i = 8000 # frame index
Frame = dlc.get_frame(Vid, i)
axes = dlc.plot_frame(Frame, axes=axes)
axes = dlc.plot_bodyparts(bodyparts, DlcDf, i, axes=axes)

# %% plot a heatmap of movement for both paws on a 2D background
fig, axes = plt.subplots()

i = 4000 # frame index
Frame = dlc.get_frame(Vid, i)
axes = dlc.plot_frame(Frame, axes=axes)
axes = dlc.plot_trajectories(DlcDf, paws, axes=axes,lw=0.025)
axes.axis('off')
axes.set_title('Whole session heatmap of paw placement')

plt.savefig(plot_dir / ('heatmap_both_paws.png'), dpi=600)

# %% Plot reaches to one side with one paw

# Settings
avg_mvmt_time = 1000
p = 0.95

# Background image
fig, axes = plt.subplots()
i = 8000 # frame index
Frame = dlc.get_frame(Vid, i)
axes = dlc.plot_frame(Frame, axes=axes)

# Detection rectangle
w = 30 # box size
rect = dlc.box2rect(SL_coords, w)
R = dlc.Rectangle(*dlc.rect2cart(rect),lw=1,facecolor='none',edgecolor='r')
axes.add_patch(R)

# Obtain all reaches within rectangle, convert from frame to time
bp = 'PR'
SpansDf = dlc.in_box_span(DlcDf, bp, rect, min_dur=1)
SpansDf = pd.DataFrame(Sync.convert(SpansDf.values,'dlc','arduino'), columns=SpansDf.columns)

# Plot all reaches to given side 
df = DlcDf[bp]
for i, row in tqdm(SpansDf.iterrows()):
    t_on = row['t_on']
    df = bhv.time_slice(DlcDf,t_on-avg_mvmt_time,t_on+500)[bp]

    ix = df.likelihood > p
    df = df.loc[ix]
    axes.plot(df.x,df.y,lw=1, alpha=0.85)

axes.set_title('Reach trajectories for right_spout with ' + str(bp))
plt.savefig(plot_dir / ('reaches_for_right_spout_with_' + str(bp) + '.png'), dpi=600)

# %% distance / speed over time
fig, axes = plt.subplots(nrows=2,sharex=True)

bps = ['PR','PL']

line_kwargs = dict(lw=1,alpha=0.8)
for i, bp in enumerate(bps):
    d_to_right = dlc.calc_dist_bp_bp(DlcDf, bp, 'SR', filter=True)
    d_to_left = dlc.calc_dist_bp_bp(DlcDf, bp, 'SR', filter=True)
    axes[i].plot(d_to_left, label='to left', **line_kwargs)
    axes[i].plot(d_to_right, label='to right', **line_kwargs)
    axes[i].set_ylabel(bp)
    axes[i].set_ylim(0)

axes[0].legend()

# %% Are they priming actions by having a specific posture for each trial type?
align_event = 'PRESENT_CUE_STATE'

fig, axes = plt.subplots(ncols=2, figsize=(5, 3))

TrialDfs_left = bhv_plt_reach.filter_trials_by(SessionDf, TrialDfs, dict(choice='left'))
TrialDfs_right = bhv_plt_reach.filter_trials_by(SessionDf, TrialDfs, dict(choice='right'))

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
Frame = dlc.get_frame(Vid, i)
axes[0] = dlc.plot_frame(Frame, axes=axes[0])
axes[1] = dlc.plot_frame(Frame, axes=axes[1])

# Formatting
axes[0].set_title('Left choice')
axes[1].set_title('Right choice')

for ax in axes:
    ax.legend(loc="center", bbox_to_anchor=(0.5, -0.2), prop={'size': 8}, ncol=2, frameon= False)
    ax.axis('off')

fig.suptitle('Paw placement aligned to ' + align_event)

plt.savefig(plot_dir / ('paw_placement_aligned_to_' + align_event + '.png'), dpi=600)

# %% Full plot of hand distance aligned to events
fig, axes = plt.subplots(nrows=2,ncols=2,sharex=True,sharey=True)

events = ['REWARD_LEFT_VALVE_ON','REWARD_RIGHT_VALVE_ON']
prev_events = ['REWARD_LEFT_AVAILABLE_EVENT','REWARD_RIGHT_AVAILABLE_EVENT']
sides = ['left','right']
coords = [SL_coords,SR_coords] # left, right
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
        D = np.zeros((np.max([ix.shape[0] for ix in inds]),len(inds)))
        D[:] = sp.nan

        # euclid dist
        dists = dlc.calc_dist_bp_point(DlcDf, bp, point, p=0.1, filter=True)

        for k in range(len(inds)):
            shape = inds[k].shape[0]
            D[:shape,k] = dists[inds[k]]

        axes[j,i].matshow(D.T,cmap='viridis_r',vmin=0,vmax=100, origin='lower', extent=(pre,post,0,D.shape[1]))

        for k in range(len(t_prev)):
            try:
                for q in t_prev[k].values:
                    axes[j,i].plot([q,q],[k,k+1],color='r', alpha=0.5,lw=1)
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
