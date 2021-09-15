#%matplotlib qt5
#%load_ext autoreload
#%autoreload 2

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 166
from matplotlib import pyplot as plt
import pandas as pd
import itertools
from pathlib import Path
import scipy as sp
import numpy as np
import seaborn as sns
import os
from tqdm import tqdm

# Custom
from Utils import behavior_analysis_utils as bhv
from Utils import metrics as met

"""
  #####  #######  #####   #####  ### ####### #     #
 #     # #       #     # #     #  #  #     # ##    #
 #       #       #       #        #  #     # # #   #
  #####  #####    #####   #####   #  #     # #  #  #
       # #             #       #  #  #     # #   # #
 #     # #       #     # #     #  #  #     # #    ##
  #####  #######  #####   #####  ### ####### #     #

"""
 
#  %% Only uses LogDf

# General functions
def plot_session_overview(SessionDf, animal_meta, session_date, axes=None):
    
    hist=10
    colors = dict(  correct="#72E043",incorrect="#F56057",premature="#9D5DF0",
                    missed="#F7D379",jackpot='#FFC0CB', anticipatory="#CB9F45")
    
    if axes is None:
        fig, axes = plt.subplots()

    for i, row in SessionDf.iterrows():
        axes.plot([i,i],[0,1],lw=2.5,color=colors[row.outcome],zorder=-1)

        w = 0.05
        if row.correct_side == 'left':
            axes.plot([i,i],[0-w,0+w],lw=1,color='k')
        if row.correct_side == 'right':
            axes.plot([i,i],[1-w,1+w],lw=1,color='k')

        if row.has_choice:
            if row.chosen_side == 'left':
                axes.plot(i,-0.0,'.',color='k')
            if row.chosen_side == 'right':
                axes.plot(i,1.0,'.',color='k')

        if row.in_corr_loop and not np.isnan(row.in_corr_loop):
            if row.correct_side == 'left':
                axes.plot([i,i],[-0.1,0.1],color='red',alpha=0.5,zorder=-2,lw=3)
            if row.correct_side == 'right':
                axes.plot([i,i],[1,1.1],color='red',alpha=0.5,zorder=-2,lw=3)
        
        if row.timing_trial and not np.isnan(row.timing_trial):
            axes.plot([i,i],[-0.1,1.1],color='cyan',alpha=0.5,zorder=-2,lw=3)

    # miss rate
    miss_rate = (SessionDf.outcome == 'missed').rolling(hist).mean()
    axes.plot(range(SessionDf.shape[0]),miss_rate,lw=1.5,color='black',alpha=0.75)
    axes.plot(range(SessionDf.shape[0]),miss_rate,lw=1,color=colors['missed'],alpha=0.75)

    # valid trials
    SDf = SessionDf.groupby('is_missed').get_group(False)
    srate = (SDf.outcome == 'correct').rolling(hist).mean()
    axes.plot(SDf.index,srate,lw=1,color='k')

    axes.axhline(0.5,linestyle=':',color='k',alpha=0.5)

    # deco
    axes.set_title(animal_meta['value'][5] + " - " + animal_meta['value'][0] + " - " + str(session_date))
    axes.set_xlabel('trial #')
    axes.set_ylabel('success rate')
    return axes

def plot_overview_simple(LogDf, align_event, pre, post, how='bars', axes=None):
    "Plots trials together with reach spans"

    if axes is None:
        fig, axes = plt.subplots()

    # Key Events and Spans
    key_list = ['REACH_LEFT_ON', 'REACH_RIGHT_ON', 'PRESENT_INTERVAL_STATE', 'GO_CUE_SHORT_EVENT', 'GO_CUE_LONG_EVENT']

    colors = sns.color_palette('hls', n_colors=len(key_list))
    cdict = dict(zip(key_list,colors))

    t_ref = bhv.get_events_from_name(LogDf, align_event).values

    for i,t in enumerate(t_ref):

        Df = bhv.time_slice(LogDf,t-pre,t+post,'t')

        for name in cdict:
            # plot events
            if name.endswith("_EVENT") or name.endswith("_STATE"):
                event_name = name
                times = bhv.get_events_from_name(Df, name).values - t
                
                if how == 'dots':
                    axes.plot(times, [i]*len(times), '.', color=cdict[event_name], alpha=0.75) # a bar
                
                if how == 'bars':
                    for time in times:
                        axes.plot([time,time],[i-0.5,i+0.5],lw=2,color=cdict[event_name], alpha=0.75) # a bar
            
            # plot spans
            if name.endswith("_ON"):
                span_name = name.split("_ON")[0]
                on_name = span_name + '_ON'
                off_name = span_name + '_OFF'

                SpansDf = bhv.get_spans_from_names(Df, on_name, off_name)

                if 'REACH' in span_name:
                    SpansDf = SpansDf[SpansDf['dt'] > 10] # remove reaches whose length is less than 10ms

                for j, row_s in SpansDf.iterrows():
                    time = row_s['t_on'] - t
                    dur = row_s['dt']
                    rect = plt.Rectangle((time,i-0.5), dur, 1, facecolor=cdict[on_name], linewidth=1)
                    axes.add_patch(rect)

    for key in cdict.keys():
        axes.plot([0],[0],color=cdict[key],label=key,lw=4)

    # Formatting
    axes.legend(loc="center", bbox_to_anchor=(0.5, -0.2), prop={'size': len(key_list)}, ncol=len(key_list), frameon=False) 
    axes.set_title('Trials aligned to ' + str(align_event))
    plt.setp(axes, xticks=np.arange(-pre, post+1, 500), xticklabels=np.arange(-pre/1000, post/1000+0.1, 0.5))
    axes.set_ylim([0, len(t_ref)])
    axes.invert_yaxis() # Needs to be after set_ylim
    axes.set_xlabel('Time (ms)')
    axes.set_ylabel('Trial No.')

    fig.tight_layout()

    return axes

def plot_overview_aligned_1st_2nd(TrialDfs, SessionDf, pre, post, axes=None): 
    """
        Session aligned to 1st cue and 2nd (timing) cues, together with choice RT markers and reach spans
        Trials split by trial outcome and interval category, similar to Gallinanes Fig.5C        
    """
    
    intervals, choice_rt = [],[]
    correct_idx, incorrect_idx, missed_idx = [],[],[]

    if axes is None:
        fig = plt.figure(constrained_layout=True, figsize=(5,4))

    # Obtain spans of ReachDf for every trial
    for TrialDf in TrialDfs:

        left_reach_Df = bhv.get_spans_from_names(TrialDf, 'REACH_LEFT_ON', 'REACH_LEFT_OFF')
        right_reach_Df = bhv.get_spans_from_names(TrialDf, 'REACH_RIGHT_ON', 'REACH_RIGHT_OFF')

        left_reach_Df = left_reach_Df[left_reach_Df['dt'] < 5] # remove reaches whose length is less than 5ms
        right_reach_Df = right_reach_Df[right_reach_Df['dt'] < 5]
    
    intervals = SessionDf['this_interval'].values + pre # to be after the pre time
    choice_rt = SessionDf['choice_rt'].values + intervals # to be after the interval

    correct_idx = np.array((SessionDf[SessionDf['outcome'] == 'correct']).index)
    incorrect_idx = np.array((SessionDf[SessionDf['outcome'] == 'incorrect']).index)
    missed_idx = np.array((SessionDf[SessionDf['outcome'] == 'missed']).index)

    # Sort the INDEXES (of data already split based on interval)
    corr_idx_sorted = correct_idx[np.argsort(intervals[correct_idx])]
    incorr_idx_sorted = incorrect_idx[np.argsort(intervals[incorrect_idx])]
    missed_idx_sorted = missed_idx[np.argsort(intervals[missed_idx])]

    split_sorted_idxs_list = [corr_idx_sorted, incorr_idx_sorted, missed_idx_sorted]

    """ Plotting """
    heights= [len(corr_idx_sorted), len(incorr_idx_sorted), len(missed_idx_sorted)]
    gs = fig.add_gridspec(ncols=1, nrows=3, height_ratios=heights)
    ylabel = ['Correct', 'Incorrect', 'Missed']

    for i, idxs in enumerate(split_sorted_idxs_list):

        axes = fig.add_subplot(gs[i]) 

        axes.set_aspect('auto')
        axes.axvline(500,linestyle='solid',alpha=1,lw=1,color='k')
        axes.axvline(2000,linestyle='dashed',alpha=0.5,lw=1,color='k')

        # Second timing cue and choice RT bars
        ymin = np.arange(-0.5,len(idxs)-1) # need to shift since lines starts at center of trial
        ymax = np.arange(0.45,len(idxs))
        axes.vlines(intervals[idxs], ymin, ymax, colors='#FFC0CB')
        axes.vlines(choice_rt[idxs], ymin, ymax, colors='#0000FF', linewidth=2)

        if i == 0:
            axes.set_title('Session overview aligned to 1st and 2nd cues') 

        axes.set_ylabel(ylabel[i])

        axes.set_xticklabels([])
        axes.set_xticks([])
        axes.set_xlim(0,post)

    # Formatting
    axes.xaxis.set_ticks_position('bottom')
    plt.setp(axes, xticks=np.arange(0, post+pre+1, 500), xticklabels=np.arange((-pre/1000), (post/1000)+0.5, 0.5))
    plt.xlabel('Time')
    
    return axes

def plot_choice_RT_hist(SessionDf, choice_interval, bin_width, axes=None):
    " Plots the choice RT histograms for 1st choice split by trial type and outcome, excludes prematures"

    sides = ['left', 'right']
    outcomes = ['correct', 'incorrect']
    
    if axes is None:
        fig, axes = plt.subplots(nrows=len(outcomes), ncols=len(sides), figsize=[4, 4], sharex=True, sharey=True)

    no_bins = round(choice_interval/bin_width)

    cmap = mpl.cm.PiYG
    colors = [cmap]

    kwargs = dict(bins = no_bins, range = (0, choice_interval), alpha=0.5, edgecolor='none')

    for i, side in enumerate(sides):
        for j, outcome in enumerate(outcomes):
            
            # Get choice rts
            try:
                SDf = SessionDf.groupby(['correct_side','outcome']).get_group((side,outcome))
                values = SDf['choice_rt'].values
            except:
                continue

            # Define color
            if (side == 'left' and outcome == 'correct') or (side == 'right' and outcome == 'incorrect'):
                color = cmap(0.1)
            else:
                color = cmap(.9)

            axes[j, i].hist(values, **kwargs, color=color, label = str([side, outcome]))
            #axes[j, i].legend(loc='upper right', frameon=False, fontsize = 8, handletextpad = 0.3, handlelength = 0.5)
        
    # Formatting
    plt.setp(axes, xticks=np.arange(0, choice_interval+1, 1000), xticklabels=np.arange(0, (choice_interval/1000)+0.1, 1))
    fig.suptitle('Choice RTs Histogram')

    for ax in axes[-1,:]:
        ax.set_xlabel('Time (s)')

    for i, ax in enumerate(axes[:,0]):
        ax.set_ylabel(outcomes[i])

    for i, ax in enumerate(axes[0,:]):
        ax.set_title(sides[i])

    fig.tight_layout()

    return axes  

# Learn to reach
def plot_reaches_between_events(SessionDf, LogDf, TrialDfs, event_1, event_2, bin_width):
    " Plots the reaches during between two events split by trial side and outcome "

    hist_range=(0,2500)

    trial_sides = ['left', 'right']
    outcomes = ['correct', 'incorrect']
    
    fig, axes = plt.subplots(nrows=len(outcomes), ncols=len(trial_sides), figsize=[6, 4], sharex=True, sharey=True)

    kwargs = dict(bins = (round(hist_range[1]/bin_width)), range = hist_range, alpha=0.5, edgecolor='none')
    
    # For each side and outcome
    for i, trial_side in enumerate(trial_sides):
        for j, outcome in enumerate(outcomes):
            
            # Filter trials (continues if there are no trials to pool from)
            try:
                TrialDfs_filt = filter_trials_by(SessionDf, TrialDfs, dict(correct_side=trial_side , outcome=outcome))
            except:
                continue
            
            # For each trial get the left and right reaches during delay period
            left_reaches, right_reaches = [],[]
            for TrialDf in TrialDfs_filt:

                t_event_1 = TrialDf[TrialDf['name'] == event_1].iloc[-1]['t'] 

                if 'GO_CUE_LEFT_EVENT' in TrialDf['name'].values: 
                    t_event_2 = TrialDf.groupby('name').get_group("GO_CUE_LEFT_EVENT").iloc[-1]['t']
                elif 'GO_CUE_RIGHT_EVENT' in TrialDf['name'].values: 
                    t_event_2 =  TrialDf.groupby('name').get_group("GO_CUE_RIGHT_EVENT").iloc[-1]['t']
                
                # Slice trials between events
                sliceDf = bhv.time_slice(LogDf, t_event_1, t_event_2)

                left_reaches.append(bhv.get_events_from_name(sliceDf, 'REACH_LEFT_ON').values-t_event_1)
                right_reaches.append(bhv.get_events_from_name(sliceDf, 'REACH_RIGHT_ON').values-t_event_1)

            flat_left_reaches = [item for sublist in left_reaches for item in sublist] 
            flat_right_reaches = [item for sublist in right_reaches for item in sublist]
               
            ax = axes[j, i]
            ax.hist(np.array(flat_left_reaches), **kwargs, label = 'reach left')
            ax.hist(np.array(flat_right_reaches), **kwargs, label = 'reach right')
            ax.legend(loc='upper right', frameon=False, fontsize = 8, handletextpad = 0.3, handlelength = 0.5)
        
    # Formatting
    axes[0, 0].set_title('short trials')
    axes[0, 1].set_title('long trials')
    axes[0, 0].set_ylabel('correct')
    axes[1, 0].set_ylabel('incorrect')
    fig.suptitle('Hist of reaches between ' + str(event_1) + ' and ' + str(event_2))
    fig.tight_layout()

    return axes

def plot_reaches_window_aligned_on_event(LogDf, align_event, pre, post, bin_width, axes=None):
    " Plots the reaches around a [-pre,post] window aligned to an event "
    
    if axes is None:
        fig, axes = plt.subplots()

    no_bins = round((post+pre)/bin_width)

    t_aligns = LogDf[LogDf['name'] == align_event]['t'] 

    left_reaches, right_reaches = np.empty([1,0]), np.empty([1,0])
    for t_align in t_aligns:

        sliceDf = bhv.time_slice(LogDf, t_align-pre, t_align + post)
        
        left_reaches = np.append(left_reaches, bhv.get_events_from_name(sliceDf, 'REACH_LEFT_ON').values - np.array(t_align))
        right_reaches = np.append(right_reaches, bhv.get_events_from_name(sliceDf, 'REACH_RIGHT_ON').values - np.array(t_align))

    # Fancy plotting
    axes.hist(left_reaches, bins = no_bins, range = (-pre,post), alpha=0.5, label = 'Left reaches')
    axes.hist(right_reaches, bins = no_bins, range = (-pre,post), alpha=0.5, label = 'Right reaches')
    plt.setp(axes, xticks=np.arange(-pre, post+1, 1000), xticklabels=np.arange(-pre/1000, post/1000+0.1, 1))
    axes.axvline(x=0, c='black')
    axes.set_xlabel('Time (s)')
    axes.set_ylabel('No. Reaches')
    axes.set_title(str(align_event))
    axes.legend()

    return axes

# Learn to choose
def plot_grasp_duration_distro(LogDf, SessionDf, bin_width, max_grasp_dur, percentile):
    " Plots the distribution of reach durations split by chosen side and outcome"
    " Works for allowing mistakes but doesnt plot text or percentile due to Nan's "

    sides = ['LEFT', 'RIGHT']

    no_bins = round(max_grasp_dur/bin_width)
    kwargs = dict(bins = no_bins, range = (0, max_grasp_dur), edgecolor='none')

    fig, axes = plt.subplots(ncols=len(sides), figsize=[6, 3], sharex=True, sharey=True)

    colors = sns.color_palette('hls', n_colors=len(sides))
    
    for ax,side,color in zip(axes,sides,colors):

        # For all reaches
        event_on, event_off = 'REACH_' + str(side) + '_ON', 'REACH_' + str(side) + '_OFF' # event names
        
        grasp_spansDf = bhv.get_spans_from_names(LogDf, event_on, event_off)
        grasp_durs = np.array(grasp_spansDf['dt'].values, dtype=object)
        ax.hist(grasp_durs, **kwargs, alpha=0.25, color = color, label = 'All grasps') # Histogram 

        # For choice-inducing reaches
        choiceDf = bhv.groupby_dict(SessionDf, dict(has_choice=True, chosen_side=side.lower()))
        grasp_durs = choiceDf[~choiceDf['grasp_dur'].isna()]['grasp_dur'].values # filter out Nans
        ax.hist(grasp_durs, **kwargs, alpha=1, color = color, label = 'Choice grasps') 

        perc = np.percentile(grasp_durs, percentile)
        ax.axvline(perc, color = color, alpha=1) # perc line
        ax.text(perc/max_grasp_dur,0.9, perc, transform = ax.transAxes) # perc number shifted a bit

    for ax,side in zip(axes,sides):
        ax.legend(frameon=False, fontsize = 'small')
        ax.set_title(side)
        ax.set_xlabel('Time (ms)')
    
    axes[0].set_ylabel('No. of occurrences')

    fig.suptitle("Histogram of grasps' duration, vertical bar is " + str(percentile) + "th percentile")    
    fig.tight_layout()

    return axes

def plot_hist_no_reaches_per_trial(LogDf, reach_crop):

    fig, axes = plt.subplots(figsize=[4, 3])
    kwargs = dict(bins = reach_crop, range = (0, reach_crop), edgecolor='none', alpha=0.5, density = True)

    # Trials spanning from choice available to end of ITI
    TrialSpans = bhv.get_spans_from_names(LogDf, "CHOICE_STATE", "TRIAL_AVAILABLE_STATE")

    TrialDfs = []
    for i, row in TrialSpans.iterrows():
        TrialDfs.append(bhv.time_slice(LogDf, row['t_on'], row['t_off']))

    no_reaches, choice_reach_idx = [],[]
    for TrialDf in TrialDfs:
        if met.has_choice(TrialDf).values:

            left_reach_spansDf = bhv.get_spans_from_names(TrialDf, 'REACH_LEFT_ON', 'REACH_LEFT_OFF')
            right_reach_spansDf = bhv.get_spans_from_names(TrialDf, 'REACH_RIGHT_ON', 'REACH_RIGHT_OFF')

            merge_spansDf = pd.concat([left_reach_spansDf, right_reach_spansDf]).reset_index(drop = True)

            choice_time = bhv.get_events_from_name(TrialDf, 'CHOICE_EVENT')['t'].values[0]
            
            # Which reach triggered the choice? The one which corresponding spansDf contains choice_time
            for i,row in enumerate(merge_spansDf.iterrows()):
                if (row[1]['t_on'] < choice_time < row[1]['t_off']):
                    choice_reach_idx.append(i+1)
            
            no_reaches.append(len(merge_spansDf))
            
    # Plotting
    axes.hist(no_reaches, color = 'r', label = 'No. reaches', **kwargs)
    axes.axvline(np.percentile(no_reaches, 75), color = 'r')
    axes.hist(choice_reach_idx, color = 'y', label = 'No. reaches \n before choice', **kwargs)
    axes.axvline(np.percentile(choice_reach_idx, 75), color = 'y')

    # Formatting
    axes.legend(frameon=False, fontsize = 'small')
    axes.set_xlabel('No. Reaches')
    axes.set_xlim([1,reach_crop]) # Can never be below 1
    axes.set_ylabel('Ratio of trials')
    axes.set_title('Histogram of no. of reaches per trial ')
    fig.tight_layout()

    return axes

# Learn to init
def plot_init_times(SessionDf, colors, axes=None):
    
    if axes is None:
        fig, axes = plt.subplots(figsize=(5, 3))

    x = range(SessionDf.shape[0])
    y = SessionDf['init_rt'] / 1000

    y_max = round(np.max(y))

    dot_colors = [colors[outcome] for outcome in SessionDf.outcome]
    axes.scatter(x , y, color=dot_colors, s=8)
    axes.axhline(0,linestyle=':',color='k',alpha=0.5)

    axes.set_title('Scatter of time to initiate trial across session')
    axes.set_ylim(-1,y_max+5) # added 5 just to avoid cropping
    axes.set_ylabel('time to init (s)')
    axes.set_xlabel('trial #')

    fig.tight_layout()

    return axes

def plot_no_init_attempts(LoadCellDf, TrialDfs, init_tresh, axes=None):
    
    if axes is None:
        fig = plt.figure(figsize=(6,4))

    sub_tresh = 0.66*init_tresh # About two thirds the original

    attempts = []
    for TrialDf in TrialDfs:

        # Slice when trial is available but not initiated
        Df = bhv.event_slice(TrialDf, "TRIAL_AVAILABLE_EVENT", "TRIAL_ENTRY_EVENT")
        
        if not Df.empty:
            LCDf = bhv.time_slice(LoadCellDf, Df.iloc[0]['t'], Df.iloc[-1]['t'])

            X = LCDf['x'].values
            Y = LCDf['y'].values

            # sub thresh pulling peaks
            X_peaks , _ = sp.signal.find_peaks(X, height = sub_tresh, width = 50) # 50 ms
            Y_peaks , _ = sp.signal.find_peaks(Y, height = sub_tresh, width = 50)

            # Computes distance between all X_peaks and all Y_peaks
            dist = np.abs(X_peaks[:, np.newaxis] - Y_peaks)

            # If its empty
            if len(dist) == 0:
                attempts.append(0)
            else:
                min_dist = np.min(dist, axis = 0)
                
                # If time difference between peaks is lower than 50 ms, they coincide in time
                no_attempts = min_dist[np.where(min_dist < 50)].shape[0]
                attempts.append(no_attempts)

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
    ax.set_ylim([0,10])
    ax.set_ylabel('No of init attempts')
    ax.set_xlabel('Trial No.')

    n, bins, _ = ax_histy.hist(attempts, bins=20, range= (0,20), density = True, orientation='horizontal')
    ax_histy.tick_params(axis="y", labelleft=False)
    ax_histy.set_xlabel('Perc. of trials (%)')
    ax_histy.set_xlim([0,1])

    ax.set_title('Number attempts before trial init')

    return ax

def CDF_of_reaches_during_delay(SessionDf,TrialDfs, axes = None, **kwargs):

    trial_types = ['short','long']

    if axes is None:
        fig, axes = plt.subplots(ncols = 2, figsize=(6, 3))

    for i, trial_type in enumerate(trial_types):

        TrialDfs_type =  filter_trials_by(SessionDf, TrialDfs, dict(interval_category=trial_type))

        rts = []
        for TrialDf_type in TrialDfs_type:
            rts.append(met.get_delay_rt(TrialDf_type))

        rts = np.array(rts)
        count, bins_count = np.histogram(rts[~np.isnan(rts)], bins = 50)
        pdf = count / len(rts) # sum of RT's since I want to include the Nans - normalize across no. trials
        cdf = np.cumsum(pdf)

        axes[i].plot(bins_count[1:], cdf, **kwargs)

        # Formatting
        axes[i].set_xlim([0,2500])
        axes[i].axhline(0.2,linestyle=':',color='r',alpha=0.25)
        axes[i].set_title(trial_type)
        axes[i].set_xlabel('Time since 1st cue (ms)')
        axes[i].set_ylim([0,1])

# Learn to time
def plot_psychometric(SessionDf, N=1000, axes=None, discrete=False):
    if axes is None:
        fig, axes = plt.subplots()

    # Get subset of timing trials with choices outside correction loops
    try:
        SDf = bhv.groupby_dict(SessionDf, dict(has_choice=True,in_corr_loop=False,timing_trial=True))
    except KeyError:
        print('No trials fulfil criteria')

    y = SDf['chosen_side'].values == 'right'
    x = SDf['this_interval'].values

    # plot the choices
    if not discrete:
        axes.plot(x,y,'.',color='k',alpha=0.5)

    axx = plt.twinx(axes)
    axx.set_yticks([0,1])
    axx.set_yticklabels(['short','long'])
    axx.set_ylabel('choice')
    w = 0.05
    axx.set_ylim(0-w, 1+w)
    axes.set_ylim(0-w, 1+w)
    axes.set_ylabel('p')
    
    axes.axvline(1500,linestyle=':',alpha=0.5,lw=1,color='k')
    axes.axhline(0.5,linestyle=':',alpha=0.5,lw=1,color='k')

    # plot the fit
    x_fit = np.linspace(0,3000,100)
    axes.plot(x_fit, bhv.log_reg(x, y, x_fit),color='red', linewidth=2,alpha=0.75)

    # plot the random models based on the choice bias
    bias = (SDf['chosen_side'] == 'right').sum() / SDf.shape[0]
    R = []
    for i in tqdm(range(N)):
        rand_choices = np.random.rand(SDf.shape[0]) < bias
        try:
            R.append(bhv.log_reg(x, rand_choices,x_fit))
        except ValueError:
            # thrown when all samples are true or false
            print("all true or false")
            pass
    R = np.array(R)

    # Several statistical boundaries
    alphas = [5, 0.5, 0.05]
    opacities = [0.2, 0.2, 0.2]
    for alpha, a in zip(alphas, opacities):
        R_pc = sp.percentile(R, (alpha, 100-alpha), 0)
        axes.fill_between(x_fit, R_pc[0], R_pc[1], color='blue', alpha=a, linewidth=0)

    axes.set_xlabel('time (ms)')

    if discrete:
        intervals = list(SessionDf.groupby('this_interval').groups.keys())
        correct_sides = ['right']*len(intervals)
        for i, interval in enumerate(intervals):
            SDf = bhv.groupby_dict(SessionDf, dict(this_interval=interval,has_choice=True,in_corr_loop=False,timing_trial=True))
            f = (SDf['chosen_side'] == correct_sides[i]).sum() / SDf.shape[0]
            axes.plot(interval,f,'o',color='r')

    return axes

def outcome_split_by_interval_category(SessionDf, axes = None):

    fig, axes = plt.subplots(figsize=(6, 4))

    colors = dict(correct="#72E043", incorrect="#F56057", premature="#9D5DF0", missed="#F7D379", jackpot = '#FFC0CB')
    bar_width = 0.35

    # labels are time intervals
    intervals = np.sort(SessionDf['this_interval'].unique())
    labels = [str(interval) for interval in intervals]

    # there are four outcome subcategories - Correct, Incorrect, Premature, Missed
    outcomes = SessionDf['outcome'].unique()

    bottom = np.zeros(len(intervals))
    # get fraction of each outcome relative to all trials grouped by interval 
    for outcome in outcomes:

        bar_fractions = []
        for interval in intervals:

            try:
                # number of trials per interval category
                interval_trials = len(SessionDf.groupby('this_interval').get_group(interval))

                # number of trials with outcome per interval category
                outcome_interval_trials = len(SessionDf.groupby(['outcome','this_interval']).get_group((outcome,interval)))
                bar_fractions.append(outcome_interval_trials/interval_trials)

            except:
                print('Zero trials for given pair: ' + str(outcome) + ',' + str(interval))
                bar_fractions.append(0)

        bar_fractions = np.array(bar_fractions)
        axes.bar(labels, bar_fractions, bar_width, bottom=bottom, label=outcome, color=colors[outcome])

        # for stacked bars
        bottom = bottom + bar_fractions # element-wise addition

    axes.legend(loc="center", frameon = False, bbox_to_anchor=(0.5, 1.1), ncol=len(colors))
    axes.set_ylabel('Fraction of trials')
    axes.set_ylim([0,1])
    axes.set_xlabel('Interval categories')
    axes.set_title('outcome split by interval category')

    return axes

# %% Uses DLC data
def caracterize_missed_trials(LogDf,LoadCellDf,DlcDf,axes=None):
    """ 
        Caraterizes missed trials into classes of increasing proximity to a successful choice:
        1 - Staying still, completely disengaged
        2 - Aware of task variables
        3 - Making prep for reach but not following through
        4 - Reaching but failing the target
        5 - Insufficient grasp duration
    """

    return axes

def plot_trajectories_with_marker(LogDf, SessionDf, labelsDf, align_event, pre, post, animal_id, axes=None):
    " Plots trajectories from align event until choice with marker"

    if axes is None:
        _ , axes = plt.subplots()

    ts_align = LogDf.loc[LogDf['name'] == align_event, 't'].values[0]

    left_paw, right_paw = [],[]

    for t_align in ts_align:
        lDf = bhv.time_slice(labelsDf, t_align-pre, t_align+post)

        #left_paw.append(lDf['x'].values)
        #right_paw.append(lDf['y'].values)


    #Fx = np.array(Fx).T
    #Fy = np.array(Fy).T

    return axes

def plot_mean_trajectories(LogDf, LoadCellDf, SessionDf, TrialDfs, align_event, pre, post, animal_id, axes=None):
    """ Plots trajectories in 2D aligned to an event """

    if axes==None:
        fig , axes = plt.subplots()

    Fx,Fy,_ = bhv.get_FxFy_window_aligned_on_event(LoadCellDf, TrialDfs, align_event, pre, post)
    F = np.dstack((Fx,Fy)) # 1st dim is time, 2nd is trials, 3rd is X/Y
    
    # time-varying color code
    cm = plt.cm.get_cmap('Blues')

    z = np.linspace(0, 1, num = F.shape[0])

    F_mean = np.mean(F,1).T
    scatter = plt.scatter(F_mean[0, :], F_mean[1, :], c=z, cmap= cm, s = 4)

    plt.clim(-0.3, 1)
    cbar = plt.colorbar(scatter, orientation='vertical', aspect=60)
    cbar.set_ticks([-0.3, 1]); cbar.set_ticklabels([str(pre/1000) + 's', str(post/1000) + 's'])

    # Formatting
    axes.axvline(0 ,linestyle=':',alpha=0.5,lw=1,color='k')
    axes.axhline(0 ,linestyle=':',alpha=0.5,lw=1,color='k')
    axes.set_xlabel('Left/Right axis')
    axes.set_ylabel('Front/Back axis')
    axes.set_title(' Mean 2D trajectories aligned to ' + str(align_event) + ' ' + str(animal_id))
    axes.legend(frameon=False, markerscale = 3)


    axes.set_xlim([-3500,3500])
    axes.set_ylim([-3500,3500])
    [s.set_visible(False) for s in axes.spines.values()]

    # Bounding box
    Y_thresh = np.mean(LogDf[LogDf['var'] == 'Y_thresh'].value.values)
    X_thresh = np.mean(LogDf[LogDf['var'] == 'X_thresh'].value.values)

    if np.isnan(X_thresh):
        X_thresh = 2500
        print('No Y_tresh update on LogDf, using default for analysis')
    if np.isnan(Y_thresh):   
        Y_thresh = 2500
        print('No Y_tresh update on LogDf, using default for analysis')
        
    axes.add_patch(patches.Rectangle((-X_thresh,-Y_thresh), 2*X_thresh, 2*Y_thresh,fill=False))

    if fig:
        fig.tight_layout() 

    return axes

# %% Uses Cam data


"""
    #    #     # ### #     #    #    #
   # #   ##    #  #  ##   ##   # #   #
  #   #  # #   #  #  # # # #  #   #  #
 #     # #  #  #  #  #  #  # #     # #
 ####### #   # #  #  #     # ####### #
 #     # #    ##  #  #     # #     # #
 #     # #     # ### #     # #     # #######

"""

def plot_sessions_overview(LogDfs, paths, task_name, animal_id, axes = None):
    " Plots trials performed together with every trial outcome plus sucess rate and weight across sessions"

    if axes is None:
        fig , axes = plt.subplots(ncols=2, sharex=True, figsize=(9, 4))

    trials_performed = []
    trials_correct = []
    trials_incorrect = []
    trials_missed = []
    trials_premature = []
    weight = []
    date = []

    # Obtaining number of trials of X
    for LogDf,path in zip(LogDfs, paths):

        # Correct date format
        folder_name = os.path.basename(path)
        complete_date = folder_name.split('_')[0]
        month = calendar.month_abbr[int(complete_date.split('-')[1])]
        day = complete_date.split('-')[2]
        date.append(month+'-'+day)

        # Total time
        session_dur = round((LogDf['t'].iat[-1]-LogDf['t'].iat[0])/60000) # convert to min

        # Total number of trials performed
        event_times = bhv.get_events_from_name(LogDf,"TRIAL_ENTRY_STATE")
        trials_performed.append(len(event_times)/session_dur)

        # Missed trials
        missed_choiceDf = bhv.get_events_from_name(LogDf,"CHOICE_MISSED_EVENT")
        trials_missed.append(len(missed_choiceDf)/session_dur)

        # Premature trials
        try:
            premature_choiceDf = bhv.get_events_from_name(LogDf,"PREMATURE_CHOICE_EVENT")
            trials_premature.append(len(premature_choiceDf)/session_dur)
        except:
            trials_premature.append(None)

        # Correct trials 
        correct_choiceDf = bhv.get_events_from_name(LogDf,'CHOICE_CORRECT_EVENT')
        trials_correct.append(len(correct_choiceDf)/session_dur)
        
        # Incorrect trials 
        incorrect_choiceDf = bhv.get_events_from_name(LogDf,'CHOICE_INCORRECT_EVENT')
        trials_incorrect.append(len(incorrect_choiceDf)/session_dur)

        # Weight
        try:
            animal_meta = pd.read_csv(path.joinpath('animal_meta.csv'))
            weight.append(round(float(animal_meta.at[6, 'value'])/float(animal_meta.at[4, 'value']),2))
        except:
            weight.append(None)

    sucess_rate = np.multiply(np.divide(trials_correct,trials_performed),100)

    # Subplot 1
    axes[0].plot(trials_performed, color = 'blue', label = 'Performed')
    axes[0].plot(trials_correct, color = 'green', label = 'Correct')
    axes[0].plot(trials_incorrect, color = 'red', label = 'Incorrect')
    axes[0].plot(trials_missed, color = 'black', label = 'Missed')
    axes[0].plot(trials_premature, color = 'pink', label = 'Premature')
    
    axes[0].set_ylabel('Trial count per minute')
    axes[0].set_xlabel('Session number')
    axes[0].legend(loc='upper left', frameon=False) 

    fig.suptitle('Sessions overview in ' + task_name + ' for mouse ' + animal_id)
    plt.setp(axes[0], xticks=np.arange(0, len(date), 1), xticklabels=date)
    plt.xticks(rotation=45)
    plt.setp(axes[0], yticks=np.arange(0, max(trials_performed), 1), yticklabels=np.arange(0,  max(trials_performed), 1))
      
    # Two sided axes Subplot 2
    axes[1].plot(sucess_rate, color = 'green', label = 'Sucess rate')
    axes[1].legend(loc='upper left', frameon=False) 
    axes[1].set_ylabel('a.u. (%)')
    plt.setp(axes[1], yticks=np.arange(0,100,10), yticklabels=np.arange(0,100,10))

    weight = np.multiply(weight,100)
    twin_ax = axes[1].twinx()
    twin_ax.plot(weight, color = 'gray')
    twin_ax.set_ylabel('Normalized Weight to max (%)', color = 'gray')
    plt.setp(twin_ax, yticks=np.arange(75,100+1,5), yticklabels=np.arange(75,100+1,5))

    fig.autofmt_xdate()
    plt.show()

    return axes


"""
 #     # ###  #####   #####
 ##   ##  #  #     # #     #
 # # # #  #  #       #
 #  #  #  #   #####  #
 #     #  #        # #
 #     #  #  #     # #     #
 #     # ###  #####   #####

"""

def filter_trials_by(SessionDf, TrialDfs, filter_dict):
    """
        This function filters input TrialDfs given filter_pair tuple (or list of tuples)
        Example: given dict(outcome='correct', chosen_side='left') it will only output trials which are correct to left side 
    """

    if len(filter_dict) == 1: # in case its only one pair
        groupby_keys = list(filter_dict.keys())
        getgroup_keys = list(filter_dict.values())[0]

        try:
            SDf = SessionDf.groupby(groupby_keys).get_group(getgroup_keys)
        except:
            print('No trials with given input filter_pair combination')
            
    else: # more than 1 pair
        try:
            SDf = bhv.groupby_dict(SessionDf, filter_dict)
        except:
            print('No trials with given input filter_pair combination')

    TrialDfs_filt = [TrialDfs[i] for i in SDf.index.values.astype(int)]

    return TrialDfs_filt

def truncate_pad_vector(arrs, pad_with = None, max_len = None):
    " Truncate and pad an array with rows of different dimensions to max_len (defined either by user or input arrs)"
    
    # In case length is not defined by user
    if max_len == None:
        list_len = [len(arr) for arr in arrs]
        max_len = max(list_len)

    if pad_with == None:
        pad_with = np.NaN
    
    trunc_pad_arr = np.empty((len(arrs), max_len)) 
    trunc_pad_arr[:] = np.NaN # Initialize with all Nans

    for i, arr in enumerate(arrs):
        if len(arr) < max_len:
            trunc_pad_arr[i,:] = np.pad(arr, (0, max_len-arr.shape[0]), mode='constant',constant_values=(pad_with,))
        elif len(arr) > max_len:
            trunc_pad_arr[i,:] = np.array(arr[:max_len])
        elif len(arr) == max_len:
            trunc_pad_arr[i,:] = np.array(arr)

    return trunc_pad_arr

def get_LC_slice_aligned_on_event(LoadCellDf, TrialDfs, align_event, pre, post):
    " Returns NUMPY ND.ARRAY of all trials aligned to an event in a window defined by [align-pre, align+post]"

    X, Y = [],[]

    for TrialDf in TrialDfs:
        
        t_align = TrialDf.loc[TrialDf['name'] == align_event, 't'].values[0]
        LCDf = bhv.time_slice(LoadCellDf, t_align-pre, t_align+post) # slice around reference event

        # Store
        X.append(LCDf['x'].values)
        Y.append(LCDf['y'].values)

    # Turn into numpy arrays
    X = np.array(X, dtype=object).T
    Y = np.array(Y, dtype=object).T

    return X,Y

def get_dist_aligned_on_event(DlcDf, TrialDfs, align_event, pre, post, func, f_arg1, f_arg2):
    """
        Returns dist for func and args for all trials aligned to an event in a window defined by [align-pre, align+post]
        NOTE: we are returning NP nd.arrays so we use advanced slicing, meaning [row, col] instead of [row][col]
    """

    dist = []
    
    for TrialDf in TrialDfs:
        # Get time point of align_event
        t_align = TrialDf.loc[TrialDf['name'] == align_event, 't'].values[0]

        # Slice DlcDf and compute distance according to input func and args
        Dlc_TrialDf = bhv.time_slice(DlcDf, t_align-pre, t_align+post)
        dist.append(func(Dlc_TrialDf,f_arg1, f_arg2, filter=True))

    # Fix the fact that some arrays have different lengths (due to frame rate fluctuations)
    dist = truncate_pad_vector(dist)

    return dist

def get_dist_between_events(DlcDf, TrialDfs, first_event, second_event, func, f_arg1, f_arg2, pad_with = None):
    """
        Returns Fx/Fy/Fmag NUMPY ND.ARRAY with dimensions (trials, max_array_len) for all trials 
        NOTE: we are returning NP nd.arrays so we use advanced slicing, meaning [row, col] instead of [row][col]

        func anf f_arg1/2 are names for a more general implementation yet to test where one can get any function 
        between two events (does not have to be distance despite the function name)
    """

    dist = []
    for i, TrialDf in enumerate(TrialDfs):
        if not TrialDf.empty:

            if first_event == 'first': # From start of trial
                time_1st = float(TrialDf['t'].iloc[0])
            else:
                time_1st = float(TrialDf[TrialDf.name == first_event]['t'])

            if second_event == 'last': # Until end of trial
                time_2nd = float(TrialDf['t'].iloc[-1])
            else:
                time_2nd = float(TrialDf[TrialDf.name == second_event]['t'])

            # Slice DlcDf and compute distance according to input func and args
            Dlc_TrialDf = bhv.time_slice(DlcDf, time_1st, time_2nd)
            dist.append(func(Dlc_TrialDf,f_arg1, f_arg2, filter=True))

    # Make sure we have numpy arrays with same length, pad/truncate with given input pad_with
    dist = bhv.truncate_pad_vector(dist, pad_with)

    return dist

def compute_choice_grasp_dur(LogDf, SessionDf):
    " FIXME - LAST CHOICE GRASP DUR IS NEVER COMPUTED SINCE THERE IS NO TRIAL AVAILABLE"

    # Trials spanning from choice available to end of ITI
    TrialSpans = bhv.get_spans_from_names(LogDf, "CHOICE_STATE", "TRIAL_AVAILABLE_STATE")

    TrialDfs = []
    for i, row in TrialSpans.iterrows():
        TrialDfs.append(bhv.time_slice(LogDf, row['t_on'], row['t_off']))

    # Compute grasp_dur for every trial
    grasp_dur = pd.Series()
    for TrialDf in TrialDfs:
        if met.has_choice(TrialDf).values:

            left_reach_spansDf = bhv.get_spans_from_names(TrialDf, 'REACH_LEFT_ON', 'REACH_LEFT_OFF')
            right_reach_spansDf = bhv.get_spans_from_names(TrialDf, 'REACH_RIGHT_ON', 'REACH_RIGHT_OFF')

            merge_spansDf = pd.concat([left_reach_spansDf, right_reach_spansDf]).reset_index(drop = True)

            choice_time = bhv.get_events_from_name(TrialDf, 'CHOICE_EVENT')['t'].values[0]
            
            # Which reach triggered the choice? The one which corresponding spansDf contains choice_time
            for row in merge_spansDf.iterrows():
                if (row[1]['t_on'] < choice_time < row[1]['t_off']):
                    grasp_dur = grasp_dur.append(pd.Series(row[1]['dt']))

        else: grasp_dur = grasp_dur.append(pd.Series(np.NaN))

    # FIXME Adding last trial manually
    grasp_dur = grasp_dur.append(pd.Series(np.NaN))

    SessionDf['grasp_dur'] = grasp_dur.reset_index(drop = True)

    return SessionDf
                