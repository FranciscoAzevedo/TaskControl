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
import scipy.signal
import numpy as np
import pandas as pd

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

h5_path = utils.get_file_dialog(initial_dir=fd_path)
DlcDf = dlc.read_dlc_h5(h5_path)
bodyparts = np.unique([j[0] for j in DlcDf.columns[1:]]) # all body parts
spouts = ['SL','SR']
paws = ['PL','PR']

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
lc_sync_event = sync.parse_harp_sync(fd_path / 'bonsai_harp_sync.csv')
arduino_sync_event = sync.get_arduino_sync(fd_path / 'arduino_log.txt')

# Get the values out of them
Sync = sync.Syncer()
Sync.data['arduino'] = arduino_sync_event['t'].values
Sync.data['loadcell'] = lc_sync_event['t'].values
Sync.data['dlc'] = cam_sync_event.index.values # the frames are the DLC
Sync.data['cam'] = cam_sync_event['t'].values # used for what?

# Sync them all to master clock (arduino [FSM?] at ~1Khz) and convert
#LoadCellDf['t_loadcell'] = LoadCellDf['t'] # keeps the original

Sync.sync('loadcell','arduino')
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

#  Plots dir and animal info
animal_meta = pd.read_csv(log_path.parent.parent / 'animal_meta.csv')
nickname = animal_meta[animal_meta['name'] == 'Nickname']['value'].values[0]
session_date = log_path.parent.stem.split('_')[0]

plot_dir = log_path.parent / 'plots'
os.makedirs(plot_dir, exist_ok=True)

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
right_spout = [230,400] # spout right
#left_spout = [380, 405] # spout left
avg_mvmt_time = 250
p = 0.99

# Background image
fig, axes = plt.subplots()
i = 8000 # frame index
Frame = dlc.get_frame(Vid, i)
axes = dlc.plot_frame(Frame, axes=axes)

# Detection rectangle
w = 75 # box size
rect = dlc.box2rect(right_spout, w)
R = dlc.Rectangle(*dlc.rect2cart(rect),lw=1,facecolor='none',edgecolor='r')
axes.add_patch(R)

# Obtain all reaches within rectangle, convert from frame to time
bp = 'PR'
SpansDf = dlc.in_box_span(DlcDf, bp, rect, min_dur=5)
SpansDf = pd.DataFrame(dlc.frame2time(SpansDf.values,m,b,m2,b2),columns=SpansDf.columns)

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
    d_to_right = bhv.calc_dist_bp_point(DlcDf, bp, right_spout, filter=True)
    d_to_left = bhv.calc_dist_bp_point(DlcDf, bp, left_spout, filter=True)
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

TrialDfs_left = bhv_plt_reach.filter_trials_by(SessionDf, TrialDfs, dict(trial_side='left'))
TrialDfs_right = bhv_plt_reach.filter_trials_by(SessionDf, TrialDfs, dict(trial_side='right'))

# General function to be applied 
func = dlc.calc_dist_bp_point

d_to_left = bhv.get_dist_aligned_on_event(DlcDf, TrialDfs_left, align_event, pre, post, func, 'PR', left_spout)
d_to_right = bhv.get_dist_aligned_on_event(DlcDf, TrialDfs_right, align_event, pre, post, func, 'PR', right_spout)

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
        dists = bhv.calc_dist_bp_point(DlcDf, bp, point, p=0.1, filter=True)

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
frame = dlc.get_frame(Vid, ix[0])
im = ax.imshow(frame, cmap='gray')
# ax, lines = plot_Skeleton(Skeleton, DlcDf, ix[0], axes=ax)

def update(i):
    Frame = dlc.get_frame(Vid,i)
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