# %% imports 
%matplotlib qt5
%load_ext autoreload
%autoreload 2

# Misc libs
import sys
import os
from pathlib import Path
from tqdm import tqdm

# Plotting libs
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import seaborn as sns

# Computational libs
import scipy as sp
import numpy as np
import pandas as pd
import cv2

sys.path.append('..')

# Custom libs
from Utils import behavior_analysis_utils as bhv
from Utils import dlc_analysis_utils as dlc
from Utils import utils
from Utils.metrics import *
from Utils import sync

"""
 #       #######    #    ######  ### #     #  #####
 #       #     #   # #   #     #  #  ##    # #     #
 #       #     #  #   #  #     #  #  # #   # #
 #       #     # #     # #     #  #  #  #  # #  ####
 #       #     # ####### #     #  #  #   # # #     #
 #       #     # #     # #     #  #  #    ## #     #
 ####### ####### #     # ######  ### #     #  #####

"""

# %% read all three data sources
fd_path = Path("/media/storage/shared-paton/georg/Animals_reaching/JJP-01975_Marquez/2021-04-29_11-16-15_learn_to_fixate_discrete_v1")
#fd_path = utils.get_folder_dialog(initial_dir="/media/storage/shared-paton/georg/")

# DLC data
try: # if filtered is present take it, otherwise default to unfiltered
    h5_path = fd_path / [fname for fname in os.listdir(fd_path) if fname.endswith('filtered.h55')][0]
except IndexError:
    h5_path = fd_path / [fname for fname in os.listdir(fd_path) if fname.endswith('.h5')][0]
DlcDf = dlc.read_dlc_h5(h5_path)

# getting all dlc body parts
bodyparts = np.unique([j[0] for j in DlcDf.columns[1:]])

# Camera data
video_path = fd_path / "bonsai_video.avi"
Vid = dlc.read_video(str(video_path))

# Arduino data
log_path = fd_path / 'arduino_log.txt'
LogDf = bhv.get_LogDf_from_path(log_path)

session_metrics = [has_choice, get_chosen_side, get_chosen_interval, get_correct_side,
                    get_interval_category, get_interval, get_outcome, get_in_corr_loop,
                    get_timing_trial, get_start, get_stop, get_init_rt, get_premature_rt,
                    get_choice_rt]

SessionDf, TrialDfs = utils.get_SessionDf(LogDf, session_metrics, "TRIAL_ENTRY_EVENT")


# LoadCell Data
# LoadCellDf = bhv.parse_bonsai_LoadCellData(fd_path / 'bonsai_LoadCellData.csv')

# %% Synching
from Utils import sync
cam_sync_event = sync.parse_cam_sync(fd_path / 'bonsai_frame_stamps.csv')
# lc_sync_event = sync.parse_harp_sync(fd_path / 'bonsai_harp_sync.csv')
arduino_sync_event = sync.get_arduino_sync(fd_path / 'arduino_log.txt')

Sync = sync.Syncer()
Sync.data['arduino'] = arduino_sync_event['t'].values
# Sync.data['loadcell'] = lc_sync_event['t'].values
Sync.data['dlc'] = cam_sync_event.index.values # the frames are the DLC
Sync.data['cam'] = cam_sync_event['t'].values # used for what?
Sync.sync('arduino','dlc')
Sync.sync('arduino','cam')

DlcDf['t'] = Sync.convert(DlcDf.index.values, 'dlc', 'arduino')

# %%
"""
.########..########..########.########..########...#######...######..########..######.
.##.....##.##.....##.##.......##.....##.##.....##.##.....##.##....##.##.......##....##
.##.....##.##.....##.##.......##.....##.##.....##.##.....##.##.......##.......##......
.########..########..######...########..########..##.....##.##.......######....######.
.##........##...##...##.......##........##...##...##.....##.##.......##.............##
.##........##....##..##.......##........##....##..##.....##.##....##.##.......##....##
.##........##.....##.########.##........##.....##..#######...######..########..######.
"""

def calc_bodypart_speed(DlcDf, bodyparts):
    for i,bp in enumerate(tqdm(bodyparts)):
        Vxy = sp.diff(DlcDf[bp][['x','y']].values,axis=0) / DlcDf['t'][:-1].values[:,sp.newaxis]
        V = sp.sqrt(sp.sum(Vxy**2,axis=1)) # euclid vector norm
        V = V / sp.diff(DlcDf['t'].values) # -> to speed

        V = sp.concatenate([[sp.nan],V]) # pad first to nan (speed undefined)
        DlcDf[(bp,'v')] = V
    return DlcDf

def interpolate_bodypart_pos(DlcDf, bodyparts, p, kind='linear', fill_value=np.NaN):
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

def make_bodypart_colors(bodyparts):
    bp_left = [bp for bp in bodyparts if bp.endswith('L')]
    bp_right = [bp for bp in bodyparts if bp.endswith('R')]
    c_l = sns.color_palette('viridis', n_colors=len(bp_left))
    c_r = sns.color_palette('magma', n_colors=len(bp_right))
    bp_cols = dict(zip(bp_left+bp_right,c_l+c_r))
    return bp_cols

# Interpolate missing labels, compute speeds and add to DlcDf
DlcDf = interpolate_bodypart_pos(DlcDf, bodyparts, p=0.99)
DlcDf = calc_bodypart_speed(DlcDf, bodyparts)

# %%
"""
 
  ######  ########    ###    ######## ####  ######  
 ##    ##    ##      ## ##      ##     ##  ##    ## 
 ##          ##     ##   ##     ##     ##  ##       
  ######     ##    ##     ##    ##     ##  ##       
       ##    ##    #########    ##     ##  ##       
 ##    ##    ##    ##     ##    ##     ##  ##    ## 
  ######     ##    ##     ##    ##    ####  ######  
 
"""

# %% selecting t_on and t_off based on trial type

# trial selection
# SDf = bhv.groupby_dict(SessionDf, dict(outcome='correct', correct_side='left'))
# TrialDf = TrialDfs[SDf.index[0]]
TrialDf = TrialDfs[0]

Df = bhv.event_slice(TrialDf, 'TRIAL_ENTRY_EVENT', 'ITI_STATE')
t_on = Df.iloc[0]['t']
t_off = Df.iloc[-1]['t']


# %% static image with trajectory between t_on and t_off

bp_cols = make_bodypart_colors(bodyparts)

fig, axes = plt.subplots()
frame_ix = Sync.convert(t_on, 'arduino', 'dlc')
frame = dlc.get_frame(Vid, frame_ix)
dlc.plot_frame(frame, axes=axes)
dlc.plot_bodyparts(bodyparts, DlcDf, frame_ix, colors=bp_cols, axes=axes)

# trajectory
DlcDfSlice = bhv.time_slice(DlcDf, t_on, t_off)
dlc.plot_trajectories(DlcDfSlice, bodyparts, axes=axes, colors=bp_cols, lw=1, p=0.99)

# %% plot all of the selected trial type
# trial selection
SDf = bhv.groupby_dict(SessionDf, dict(correct_side='left',outcome='incorrect'))
# SDf = SessionDf.groupby('correct_side').get_group('right')

# plot some random frame
fig, axes = plt.subplots()
frame_ix = 1000
frame = dlc.get_frame(Vid, frame_ix)
dlc.plot_frame(frame, axes=axes)

# plot all traj in selection
for i in tqdm(SDf.index):
    TrialDf = TrialDfs[i]
    Df = bhv.event_slice(TrialDf,'PRESENT_INTERVAL_STATE','ITI_STATE')
    t_on = Df.iloc[0]['t']
    t_off = Df.iloc[-1]['t']

    # trial by trial colors
    bp_cols = {}
    cmaps = dict(zip(bodyparts,['viridis','magma']))
    for bp in bodyparts:
        c = sns.color_palette(cmaps[bp],as_cmap=True)(sp.rand())
        bp_cols[bp] = c

    # marker for the start
    # frame_ix = dlc.time2frame(t_on, m, b, m2, b2)
    frame_ix = Sync.convert(t_on, 'arduino', 'dlc').round().astype('int')
    dlc.plot_bodyparts(bodyparts, DlcDf, frame_ix, colors=bp_cols, axes=axes, markersize=5)

    # the trajectory
    DlcDfSlice = bhv.time_slice(DlcDf, t_on, t_off)
    dlc.plot_trajectories(DlcDfSlice, bodyparts, colors=bp_cols, axes=axes, lw=0.75, alpha=0.75, p=0.8)

# %% 
"""
 
 ##     ## #### ########  ########  #######  
 ##     ##  ##  ##     ## ##       ##     ## 
 ##     ##  ##  ##     ## ##       ##     ## 
 ##     ##  ##  ##     ## ######   ##     ## 
  ##   ##   ##  ##     ## ##       ##     ## 
   ## ##    ##  ##     ## ##       ##     ## 
    ###    #### ########  ########  #######  
 
"""

# %% Settings
import matplotlib as mpl
from matplotlib.animation import FFMpegWriter

mpl.rcParams['animation.ffmpeg_path']="/usr/local/bin/ffmpeg"

pre,post = 100, 500
save = True

key_values = [('outcome','missed'), ('correct_side','left')]

trial_idxs = bhv.groupby_dict(SessionDf, dict(key_values)).index.values

# %% Display video of trials
t_ons = SessionDf.iloc[trial_idxs]['t_on'] - pre
t_offs = SessionDf.iloc[trial_idxs]['t_off'] + post

LogDfSlice, DlcDfSlice = [],[]
for t_on, t_off in zip(t_ons,t_offs):
    LogDfSlice.append(bhv.time_slice(LogDf, t_on, t_off))
    DlcDfSlice.append(bhv.time_slice(DlcDf, t_on, t_off))

LogDfSlice = pd.concat(LogDfSlice)
DlcDfSlice = pd.concat(DlcDfSlice)
ix = list(DlcDfSlice.index)

# plotting
fig, ax = plt.subplots()
ax.axis('off')

# what events to display
display_events = list(LogDfSlice.name.unique())
if sp.nan in display_events:
    display_events.remove(sp.nan)

# image
ax.set_aspect('equal')
frame = dlc.get_frame(Vid, ix[0])
im = ax.imshow(frame, cmap='gray')

# body parts
bp_left = [bp for bp in bodyparts if bp.endswith('L')]
bp_right = [bp for bp in bodyparts if bp.endswith('R')]
c_l = sns.color_palette('viridis', n_colors=len(bp_left))
c_r = sns.color_palette('magma', n_colors=len(bp_right))
bp_cols = dict(zip(bp_left+bp_right,c_l+c_r))

bp_markers = []
for i, bp in enumerate(bodyparts):
    marker, = ax.plot([],[], 'o', color=bp_cols[bp], markersize=10)
    bp_markers.append(marker)

# traces
from matplotlib.collections import LineCollection
n_segments = 10
trace_len = 3
lws = np.linspace(0,5,n_segments)
bp_traces = []
for i, bp in enumerate(bodyparts):
    segments = []
    for j in range(n_segments):
        segment = np.zeros((trace_len,2))
        segments.append(segment)
    
    lc = LineCollection(np.array(segments),linewidths=lws,color=bp_cols[bp], alpha=0.75)
    bp_traces.append(lc)
    ax.add_artist(lc)
p = 0.0

# frame text
inactive_color = 'white'
frame_counter = ax.text(5, frame.shape[0]-25, '', color=inactive_color)
time_counter = ax.text(5, frame.shape[0]-5, '', color=inactive_color)

# event text annotations
# color setup
c = sns.color_palette('husl', n_colors=len(display_events))
event_colors = dict(zip(display_events,c))
event_display_dur = 250 # ms

event_texts = []
event_times = []
for i, event in enumerate(display_events):
    # times 
    times = LogDfSlice.groupby('name').get_group(event)['t'].values
    event_times.append(times)

    # plot
    text = ax.text(10, i*20 + 20, event, color=inactive_color, fontweight='bold', fontsize=6)
    event_texts.append(text)

fig.tight_layout()


def update(i):
    Frame = dlc.get_frame(Vid,i)
    im.set_data(Frame)

    # frame counter
    frame_counter.set_text("frame: %i - %i/%i" % (i, ix.index(i),len(ix)))
    t_abs = Sync.convert(i,'dlc','arduino') / 1000
    m = Sync.pairs[('dlc','arduino')][0]
    t_rel = (ix.index(i) * m) / 1000
    time_counter.set_text("time: %.2f - %.2f/%.2f" % (t_abs, t_rel,len(ix)*m/1000))

    # body parts
    for j, bp in enumerate(bodyparts):
        data = DlcDfSlice[bp].loc[i]
        if data['likelihood'] > p:
            bp_markers[j].set_data(data['x'], data['y'])
        else:
            bp_markers[j].set_data(sp.nan, sp.nan)
    
    # trace
    for j, bp in enumerate(bodyparts):
        i0 = i - n_segments*trace_len
        data = DlcDfSlice[bp].loc[i0:i]
        data.loc[data['likelihood'] < p] = sp.nan
        data = data[['x','y']].values[::-1,:]
        segments = bp_traces[j].get_segments()
        for k in range(n_segments):
            try:
                segments[-k] = data[k*trace_len-5:(k+1)*trace_len+5,:]
            except:
                pass
        bp_traces[j].set_segments(segments)

    for j, event in enumerate(display_events):
        t = Sync.convert(i, 'dlc', 'arduino')
        if np.any(np.logical_and(t > event_times[j], t < event_times[j] + event_display_dur)):
            event_texts[j].set_color(event_colors[event])
        else:
            event_texts[j].set_color(inactive_color)

    return (im, frame_counter, time_counter) + tuple(bp_markers) + tuple(bp_traces) + tuple(event_texts)

ani = FuncAnimation(fig, update, frames=ix, blit=True, interval=1)

if save == True:
    plot_dir = fd_path / 'videos'
    os.makedirs(plot_dir, exist_ok=True)

    Writer = FFMpegWriter(fps=20, codec='h264', extra_args=['-pix_fmt','yuv420p'])
    ani.save(plot_dir / ('10x_slow.mp4'), writer=Writer)

# %%
