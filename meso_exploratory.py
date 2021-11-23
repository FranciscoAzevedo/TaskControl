# %% imports 
%matplotlib qt5
%matplotlib qt5
%load_ext autoreload
%autoreload 2

# Plotting
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

# Computational
import scipy as sp
import numpy as np
import pandas as pd

# Misc
import os
from pathlib import Path
from tqdm import tqdm

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

# Get the values out of them
Sync = sync.Syncer()
Sync.data['arduino'] = arduino_sync_event['t'].values
Sync.data['loadcell'] = lc_sync_event['t'].values
Sync.data['dlc'] = cam_sync_event.index.values # the frames are the DLC
#Sync.data['cam'] = cam_sync_event['t'].values # used for what?

# Sync them all to one clock
Sync.sync('arduino','loadcell')

# Add single GO_CUE_EVENT
LogDf = bhv.add_go_cue_LogDf(LogDf)

#  Create SessionDf - For LEARN_TO_CHOOSE onwards
LogDf = bhv.add_go_cue_LogDf(LogDf)

session_metrics = ( met.get_start, met.get_stop, met.get_correct_side, met.get_interval_category, met.get_outcome, 
            met.get_chosen_side, met.has_reach_left, met.has_reach_right, met.get_in_corr_loop,  
            met.reach_rt_left, met.reach_rt_right, met.has_choice, met.get_interval, met.get_timing_trial,
            met.get_choice_rt, met.get_reached_side, met.get_bias, met.get_init_rt, met.rew_collected)
             

SessionDf, TrialDfs = utils.get_SessionDf(LogDf, session_metrics, "TRIAL_ENTRY_EVENT", "ITI_STATE")

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
.########..##........######.
.##.....##.##.......##....##
.##.....##.##.......##......
.##.....##.##.......##......
.##.....##.##.......##......
.##.....##.##.......##....##
.########..########..######.
"""

# DeepLabCut data and settings
try:
    h5_path = log_path.parent  / [fname for fname in os.listdir(log_path.parent) if fname.endswith('filtered.h55')][0]
except IndexError:
    h5_path = log_path.parent  / [fname for fname in os.listdir(log_path.parent) if fname.endswith('.h5')][0]

DlcDf = dlc_utils.read_dlc_h5(h5_path)
bodyparts = np.unique([j[0] for j in DlcDf.columns[1:]]) # all body parts
paws = ['PL','PR']
spouts = ['SL','SR']

# Synching 
Sync.sync('arduino','dlc')
DlcDf['t'] = Sync.convert(DlcDf.index.values, 'dlc', 'arduino')

def interpolate_bodypart_pos(DlcDf, bodyparts, p, kind='linear', fill_value='extrapolate'):
    """ interpolates x and y positions for bodyparts where likelihood is below p """
    for bp in tqdm(bodyparts):
        good_inds = DlcDf[bp]['likelihood'].values > p
        ix = DlcDf[bp].loc[good_inds].index

        bad_inds = DlcDf[bp]['likelihood'].values < p
        bix = DlcDf[bp].loc[bad_inds].index

        x = DlcDf[bp].loc[good_inds]['x'].values
        interp = sp.interpolate.interp1d(ix, x, kind=kind, fill_value=fill_value)
        DlcDf[(bp,'x')].loc[bix] = interp(bix)

        y = DlcDf[bp].loc[good_inds]['y'].values
        interp = sp.interpolate.interp1d(ix, y, kind=kind, fill_value=fill_value)
        DlcDf[(bp,'y')].loc[bix] = interp(bix)
    return DlcDf

DlcDf = interpolate_bodypart_pos(DlcDf, bodyparts, p=0.95)

# %%
"""
.########..########.##.....##....###....##.....##.####..#######..########.
.##.....##.##.......##.....##...##.##...##.....##..##..##.....##.##.....##
.##.....##.##.......##.....##..##...##..##.....##..##..##.....##.##.....##
.########..######...#########.##.....##.##.....##..##..##.....##.########.
.##.....##.##.......##.....##.#########..##...##...##..##.....##.##...##..
.##.....##.##.......##.....##.##.....##...##.##....##..##.....##.##....##.
.########..########.##.....##.##.....##....###....####..#######..##.....##
"""

# Grasp duration distro split by outcome and choice
bin_width = 5 # ms
max_grasp_dur = 250 # ms

perc = 25 #th 

bhv_plt_reach.plot_grasp_duration_distro(LogDf, SessionDf, bin_width, max_grasp_dur, perc)
plt.savefig(plot_dir / ('hist_grasp_dur_' + str(perc) + '_percentile.png'), dpi=600)

# %%
"""
.##....##.########.##.....##.########...#######..##....##..######.
.###...##.##.......##.....##.##.....##.##.....##.###...##.##....##
.####..##.##.......##.....##.##.....##.##.....##.####..##.##......
.##.##.##.######...##.....##.########..##.....##.##.##.##..######.
.##..####.##.......##.....##.##...##...##.....##.##..####.......##
.##...###.##.......##.....##.##....##..##.....##.##...###.##....##
.##....##.########..#######..##.....##..#######..##....##..######.
"""

# %% Load meso data already processed from caiman
neural_data_path = utils.get_folder_dialog(initial_dir="/media/storage/shared-paton/georg/mesoscope_testings")

neural_data = np.load(neural_data_path / "dff_cnm2_c.npy")
neuron_coords = np.load(neural_data_path / "coords.npy") # XY coords of each neuron 

# Frame acquisition events in its own index plus original one
meso_framesDf = LogDf[LogDf['name']=='FRAME_EVENT']
meso_framesDf['log_idx'] = meso_framesDf.index
meso_framesDf = meso_framesDf.reset_index(drop=True)

# %% Sanity check and interpolate missing frame events
frame2frame_diff = np.diff(meso_framesDf['t'])
main_freq = round(np.mean(frame2frame_diff))
tol = main_freq*0.05 # 5% of main freq

# Frame to frame fluctuations hist (around main freq)
fig, axes = plt.subplots()
axes.hist(frame2frame_diff, bins=round(tol*2), range=(main_freq-tol,main_freq+tol))
axes.set_ylabel('No. of ocurrences')
axes.set_xlabel('Interval between frames (ms)')
print('Diff of frame events (neural_data-log)): ' + str(neural_data.shape[1]-len(meso_framesDf)))

# identify missing frames
missing_idxs = np.array(meso_framesDf[:-1].iloc[np.abs(frame2frame_diff) > main_freq + tol].index)

# create entries corresponding to missing frames in LogDf
for missing_idx in missing_idxs:
    meso_framesDf.loc[missing_idx+0.5] = meso_framesDf.loc[missing_idx] # add event 
    meso_framesDf.loc[missing_idx+0.5, 't'] = np.nan # make t a missing value to be interpolated

meso_framesDf = meso_framesDf.sort_index().reset_index(drop=True) # reset index only after

meso_framesDf['t'] = meso_framesDf['t'].interpolate(method='linear')

# If they are still missmatched, cut the end of the mesoDf
if len(meso_framesDf) > neural_data.shape[1]:
    n = len(meso_framesDf) - neural_data.shape[1]
    meso_framesDf.drop(meso_framesDf.tail(n).index,inplace=True) 

# %% AVG across neurons across events
events = ['GO_CUE_EVENT','REWARD_STATE','REWARD_RIGHT_COLLECTED_EVENT']

fig, axes = plt.subplots(ncols=len(events), sharey=True, figsize=(8,4))

pre, post = 500,2000

for i,event in enumerate(events):
    t_refs = bhv.get_events_from_name(LogDf, event).values

    N_mean,frame_time = [],[]
    for t_ref in t_refs:

        sliceDf = bhv.time_slice(meso_framesDf, t_ref-pre, t_ref+post)
        frame_time.append(sliceDf['t'].values - t_ref) # relative to t_ref
        frame_idx = sliceDf.index.values # get frames

        N_mean.append(np.mean(neural_data[:,frame_idx], axis = 0)) # Avg across neurons

    N_mean_time = bhv.tolerant_mean(N_mean)
    time = np.arange(-pre,post+1,main_freq) # begin,stop,step
    axes[i].plot(time,N_mean_time,'o-', color = 'black')
    axes[i].vlines(0,0,1, color = 'black', alpha = 0.25)

    axes[i].set_ylim([0,0.03])
    axes[i].set_title(event)
    axes[i].set_xlabel('Time (ms)')
    axes[i].set_ylabel('\u0394F/F')

fig.suptitle('Neuronal activity avg across neurons across events aligned to')
fig.tight_layout()

# %% Stacked yplot with event ticks
fig, axes = plt.subplots(figsize=(8,8))

n_neurons = neural_data.shape[0]
n_timepoints = neural_data.shape[1]

# max min of normalized deltaF/F
dmin = np.min(neural_data)
dmax = np.max(neural_data)
dr = (dmax - dmin) * 0.7  # Crowd them a bit.

time = meso_framesDf['t']/1000

# data traces
for n_neuron in np.arange(0,n_neurons): 
    axes.plot(time,neural_data[n_neuron,:]+n_neuron*dr, color='k', linewidth=0.5)

# event traces
events = ['REACH_LEFT_ON','REACH_RIGHT_ON', 'REWARD_COLLECTED_EVENT']
colors = sns.color_palette(palette='turbo',n_colors=len(events))

for i,event in enumerate(events):
    t_refs = np.array(bhv.get_events_from_name(LogDf,event)['t'].values)
    axes.vlines(t_refs/1000, 0, n_neurons*dr, linewidth = 0.5, color=colors[i], label = event)

fig.suptitle('Neuronal activity for every neuron across sess')
axes.legend(loc="center", bbox_to_anchor=(0.5, -0.1), prop={'size': 10}, ncol=len(events), frameon=False)
axes.set_xlabel('No. neurons')
axes.set_xlabel('Time (s)')
fig.tight_layout()

# %% Clustering into groups of neurons that might be doing similar things
fig, axes = plt.subplots()

# Find best number of clusters
for i in np.arange(2,25):
    clusters, distortion = sp.cluster.vq.kmeans(neural_data, i, iter = 10)
    axes.scatter(i,distortion, color = 'b')

axes.set_ylabel('Distortion')
axes.set_xlabel('No of clusters')
axes.set_title('Distortion error across no. of clusters')

# 10 clusters seems to do it
n_clusters = 10
clusters, _ = sp.cluster.vq.kmeans(neural_data, n_clusters, iter = 10)

fig, axes = plt.subplots(figsize=(8,2))
axes.plot(clusters.T, alpha = 0.5, linewidth = 0.5)
axes.legend(frameon= False)
axes.set_title("Timecourse of kmeans' clusters")

# Now check where these clusters are in the brain
code , _ = sp.cluster.vq.vq(neural_data, clusters) # states which neuron belongs to each cluster

fig, axes = plt.subplots(figsize=(4,3))
for i in np.arange(0,n_clusters+1):
    axes.scatter(i, np.sum(code == i))
axes.set_ylabel('No. neurons')
axes.set_xlabel('No. of clusters')
axes.set_title('How many neurons belong to each cluster')
fig.tight_layout()

# plots image of brain
neural_vid_path = neural_data_path / "reconstructed.avi"
neural_vid = dlc_utils.read_video(str(neural_vid_path))

fig, axes = plt.subplots(figsize=(5,5))
frame_ix = 10
frame = dlc_utils.get_frame(neural_vid, frame_ix)
dlc_utils.plot_frame(frame, axes=axes)

# for each cluster plot pixel color coded by cluster
for i in np.arange(0,n_clusters+1):
    cluster_idxs = code == i
    axes.scatter(   neuron_coords[cluster_idxs,0],neuron_coords[cluster_idxs,1], s = 1,
                    alpha = 0.75, label = i)

axes.legend(frameon=False, bbox_to_anchor=(1,1), title = 'Cluster No.')
fig.tight_layout()

# %% Defining groups based on brain areas

# %% Neurons significantly modulated by events
