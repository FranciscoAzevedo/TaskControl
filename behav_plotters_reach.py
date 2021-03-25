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

# Custom
import behavior_analysis_utils as bhv

"""
  #####  #######  #####   #####  ### ####### #     #
 #     # #       #     # #     #  #  #     # ##    #
 #       #       #       #        #  #     # # #   #
  #####  #####    #####   #####   #  #     # #  #  #
       # #             #       #  #  #     # #   # #
 #     # #       #     # #     #  #  #     # #    ##
  #####  #######  #####   #####  ### ####### #     #

"""
 
# Only uses LogDf
def plot_session_overview(LogDf, align_event, pre, post, how='bars', axes=None):
    "Plots trials Fig5C of Gallinares"

    if axes is None:
        fig, axes = plt.subplots()

    # Key Events and Spans
    key_list = ['REACH_LEFT_ON', 'REACH_RIGHT_ON', 'PRESENT_CUE_STATE', 'PRESENT_INTERVAL_STATE']

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
                    SpansDf = SpansDf[SpansDf['dt'] > 15] # remove reaches whose length is less than 15ms

                for j, row_s in SpansDf.iterrows():
                    time = row_s['t_on'] - t
                    dur = row_s['dt']
                    rect = plt.Rectangle((time,i-0.5), dur, 1, facecolor=cdict[on_name], linewidth=2)
                    axes.add_patch(rect)

    for key in cdict.keys():
        axes.plot([0],[0],color=cdict[key],label=key,lw=4)

    # Formatting
    axes.legend(loc="center", bbox_to_anchor=(0.5, -0.2), prop={'size': 6}, ncol=len(key_list)) 
    axes.set_title('Trials aligned to ' + str(align_event))
    axes.set_ylim([0, len(t_ref)])
    axes.invert_yaxis() # Needs to be after set_ylim
    axes.set_xlabel('Time (ms)')
    axes.set_ylabel('Trial No.')

    fig.tight_layout()

    return axes

def plot_success_rate(LogDf, SessionDf, history, axes=None): 
    " Plots success rate with trial type and choice tickmarks "

    fig, axes = plt.subplots(nrows=2,gridspec_kw=dict(height_ratios=(0.1,1)))

    colors = dict(correct="#72E043", incorrect="#F56057", missed="#F7D379")

    # bars
    for i, row in SessionDf.iterrows():
        outcome = row['outcome']
        x = i
        y = 0
        rect = mpl.patches.Rectangle((x,y),1,1, edgecolor='none', facecolor=colors[outcome])
        axes[0].add_patch(rect)

    axes[0].set_xlim(0.5, SessionDf.shape[0]+0.5)
    axes[0].set_title('outcomes')
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].set_yticks([])
    axes[0].set_xticklabels([])

    x = SessionDf.index.values+1
    line_width = 0.04

    # L/R trial types
    left_trials = SessionDf.loc[SessionDf['correct_side'] == 'left'].index + 1
    y_left_trials = np.zeros(left_trials.shape[0]) - line_width
    right_trials = SessionDf.loc[SessionDf['correct_side'] == 'right'].index + 1
    y_right_trials = np.zeros(right_trials.shape[0]) + 1 + line_width

    # Some groupby keys dont always return a non-empty Df
    try:
        SDf = SessionDf.groupby('choice').get_group('left')
        left_choices = SDf.index.values+1
        y_left_choices = np.zeros(left_choices.shape[0])
        axes[1].plot(left_choices, y_left_choices, '|', color='m', label = 'left choice')
    except:
        pass

    try:
        SDf = SessionDf.groupby('choice').get_group('right')
        right_choices = SDf.index.values+1
        y_right_choices = np.zeros(right_choices.shape[0]) + 1
        axes[1].plot(right_choices, y_right_choices, '|', color='green', label = 'right choice')
    except:
        pass

    try:
        in_corr_loop = SessionDf.loc[SessionDf['in_corr_loop'] == True].index + 1
        y_in_corr_loop = np.zeros(in_corr_loop.shape[0]) + 1 + 2*line_width
        axes[1].plot(in_corr_loop, y_in_corr_loop, '|', color='r', label = 'corr. loops')
    except:
        pass

    # Grand average rate success rate
    y_sucess_rate = np.cumsum(SessionDf['outcome'] == 'correct') / (SessionDf.index.values+1)
    #y_trial_prob = (SessionDf['correct_side'] == 'right').rolling(10).mean()

    # Plotting in the same order as computed
    axes[1].plot(left_trials, y_left_trials, '|', color='k')
    axes[1].plot(right_trials, y_right_trials, '|', color='k')
    axes[1].plot(x,y_sucess_rate, color='C0', label = 'grand average')
    #axes[1].plot(x,y_trial_prob, color='k',alpha=0.3, label = 'R side (%)')

    if history is not None:
        y_filt = (SessionDf['outcome'] == 'correct').rolling(history).mean()
        axes[1].plot(x,y_filt, color='C0',alpha=0.3, label = 'rolling mean')
    
    axes[1].set_ylabel('Success rate')
    axes[1].set_xlabel('trial #')
    axes[1].set_xlim([0, len(SessionDf)])
    axes[1].legend(bbox_to_anchor=(0., 1, 0.95, .102), loc="upper center", handletextpad = 0.1, \
                frameon=False, mode = "expand", ncol = 5, borderaxespad=0., handlelength = 0.5, labelspacing = 0.8)

    return axes

def plot_choice_RT_hist(SessionDf, choice_interval, bin_width):
    " Plots the choice RT histograms for 1st choice split by trial type and outcome "

    choices = ['left', 'right']
    outcomes = ['correct', 'incorrect']
    
    fig, axes = plt.subplots(nrows=len(outcomes), ncols=len(choices), figsize=[4, 4], sharex=True, sharey=True)

    no_bins = round(choice_interval/bin_width)

    kwargs = dict(bins = no_bins, range = (0, choice_interval), alpha=0.5, edgecolor='none')

    for i, choice in enumerate(choices):
        for j, outcome in enumerate(outcomes):

            try:
                SDf = SessionDf.groupby(['choice', 'outcome']).get_group((choice, outcome))
            except:
                continue
            
            ax = axes[j, i]

            # Merge the two Dfs because either way SDf already has only the trials of interest
            rt_leftDf = SDf['choice_rt_left']
            rt_rightDf = SDf['choice_rt_right']
            df = pd.concat([rt_leftDf, rt_rightDf]).groupby(level=0).min()

            choice_rts = df.values
            ax.hist(choice_rts, **kwargs, label = str([choice, outcome]))
            ax.legend(loc='upper right', frameon=False, fontsize = 8, handletextpad = 0.3, handlelength = 0.5)
        
    # Formatting
    plt.setp(axes, xticks=np.arange(0, choice_interval+1, 1000), xticklabels=np.arange(0, (choice_interval/1000)+0.1, 1))
    fig.suptitle('Choice RTs Histogram')
    axes[0, 0].set_title('left')
    axes[0, 1].set_title('right')
    axes[0, 0].set_ylabel('correct')
    axes[1, 0].set_ylabel('incorrect')

    for ax in axes[-1,:]:
        ax.set_xlabel('Time (s)')
    fig.tight_layout()

    return axes  

def plot_psychometric(SessionDf, axes=None):
    " Timing task classic psychometric fit to data"

    if axes is None:
        axes = plt.gca()

    # get only the subset with choices
    SDf = SessionDf.groupby('has_choice').get_group(True)
    y = SDf['choice'].values == 'right'
    x = SDf['this_interval'].values

    # plot choices
    axes.plot(x,y,'.',color='k',alpha=0.5)
    axes.set_yticks([0,1])
    axes.set_yticklabels(['short','long'])
    axes.set_ylabel('choice')
    axes.axvline(1500,linestyle=':',alpha=0.5,lw=1,color='k')

    x_fit = np.linspace(0,3000,100)
    line, = plt.plot([],color='red', linewidth=2,alpha=0.75)
    line.set_data(x_fit, bhv.log_reg(x, y, x_fit))
    
    try:
        # %% random margin - with animal bias
        t = SDf['this_interval'].values
        bias = (SessionDf['choice'] == 'right').sum() / SessionDf.shape[0] # This includes premature choices now!
        R = []
        for i in range(100):
            rand_choices = np.rand(t.shape[0]) < bias # can break here if bias value is too low
            R.append(bhv.log_reg(x, rand_choices,x_fit))
        R = np.array(R)

        # Several statistical boundaries (?)
        alphas = [5, 0.5, 0.05]
        opacities = [0.5, 0.4, 0.3]
        for alpha, a in zip(alphas, opacities):
            R_pc = sp.percentile(R, (alpha, 100-alpha), 0)
            # plt.plot(x_fit, R_pc[0], color='blue', alpha=a)
            # plt.plot(x_fit, R_pc[1], color='blue', alpha=a)
            plt.fill_between(x_fit, R_pc[0], R_pc[1], color='blue', alpha=a)
        plt.set_cmap
    except KeyError:
        print('Bias too high')

    plt.setp(axes, xticks=np.arange(0, 3000+1, 500), xticklabels=np.arange(0, 3000//1000 +0.1, 0.5))
    axes.set_xlabel('Time (s)')

    return axes

# Uses DLC data


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

def plot_timing_overview(LogDf, LoadCellDf, TrialDfs, axes=None): 
    """
        Heatmap aligned to 1st cue with 2nd (timing) cue and choice RT markers, split by trial outcome and trial type
    """

    pre, post = 500, 5000
    Fx, interval, choice_RT = [],[],[]
    correct_idx, incorrect_idx, pre_idx, missed_idx = [],[],[],[]

    if axes is None:
        fig = plt.figure(constrained_layout=True)

    # for every trial initiation
    i = 0
    for TrialDf in TrialDfs:
        time_1st = float(TrialDf[TrialDf.name == 'FIRST_TIMING_CUE_EVENT']['t'])

        F = bhv.time_slice(LoadCellDf, time_1st-pre, time_1st+post)
        if (len(F) < post-pre):
            print('LCDf is shorter than LogDf!')
            continue

        Fx.append(F['x'])
        
        # Store indexes for different types of trials
        if bhv.get_outcome(TrialDf).values[0] == 'correct':
            correct_idx.append(i)
        if bhv.get_outcome(TrialDf).values[0] == 'incorrect':
            incorrect_idx.append(i)
        if bhv.get_outcome(TrialDf).values[0] == 'premature':
            pre_idx.append(i)            
        if bhv.get_outcome(TrialDf).values[0] == 'missed':
            missed_idx.append(i)

        # Store information
        interval.append(int(bhv.get_interval(TrialDf)))
        choice_RT.append(float(bhv.choice_RT(TrialDf)))

        i = i + 1

    # Ugly and hacky way to do what I want
    interval = np.array(interval) + pre
    choice_RT = np.array(choice_RT) + interval
    correct_idx = np.array(correct_idx)
    incorrect_idx = np.array(incorrect_idx)
    pre_idx = np.array(pre_idx)
    missed_idx = np.array(missed_idx)
    Fx = np.array(Fx)

    # Sort the INDEXES (of data already split based on interval)
    corr_idx_sorted = correct_idx[np.argsort(interval[correct_idx])]
    incorr_idx_sorted = incorrect_idx[np.argsort(interval[incorrect_idx])]
    pre_idx_sorted = pre_idx[np.argsort(interval[pre_idx])]
    missed_idx_sorted = missed_idx[np.argsort(interval[missed_idx])]

    split_sorted_idxs_list = [corr_idx_sorted, incorr_idx_sorted, pre_idx_sorted, missed_idx_sorted]

    """ Plotting """
    heights= [len(corr_idx_sorted), len(incorr_idx_sorted), len(pre_idx_sorted), len(missed_idx_sorted)]
    gs = fig.add_gridspec(ncols=1, nrows=4, height_ratios=heights)
    ylabel = ['Correct', 'Incorrect', 'Premature', 'Missed']

    for i, idxs in enumerate(split_sorted_idxs_list):

        axes = fig.add_subplot(gs[i]) 
        force_x_tresh = 2500
        heat = axes.matshow(Fx[idxs,:], cmap='RdBu',vmin=-force_x_tresh,vmax=force_x_tresh) # X axis
        axes.set_aspect('auto')
        axes.axvline(500,linestyle='solid',alpha=0.5,lw=1,color='k')
        axes.axvline(2000,linestyle='solid',alpha=0.25,lw=1,color='k')

        # Second timing cue and choice RT bars
        ymin = np.arange(-0.5,len(idxs)-1) # need to shift since lines starts at center of trial
        ymax = np.arange(0.45,len(idxs))
        axes.vlines(interval[idxs], ymin, ymax, colors='k', alpha=0.75)
        axes.vlines(choice_RT[idxs], ymin, ymax, colors='#7CFC00', linewidth=2)

        if i == 0:
            axes.set_title('Forces X axis aligned to 1st timing cue') 

        axes.set_ylabel(ylabel[i])

        axes.set_xticklabels([])
        axes.set_xticks([])
        axes.set_xlim(0,5500)

    # Formatting
    axes.xaxis.set_ticks_position('bottom')
    plt.setp(axes, xticks=np.arange(0, post+pre+1, 500), xticklabels=np.arange((-pre/1000), (post/1000)+0.5, 0.5))
    plt.xlabel('Time')
    
    cbar = plt.colorbar(heat, orientation='horizontal', aspect = 50)
    cbar.set_ticks([-2000,-1000,0,1000,2000]); cbar.set_ticklabels(["Left (-2000)","-1000","0","1000","Right (2000)"])

    return axes

# Uses Video

"""
    #    #     # ### #     #    #    #
   # #   ##    #  #  ##   ##   # #   #
  #   #  # #   #  #  # # # #  #   #  #
 #     # #  #  #  #  #  #  # #     # #
 ####### #   # #  #  #     # ####### #
 #     # #    ##  #  #     # #     # #
 #     # #     # ### #     # #     # #######

"""

# Number of responded trials (%)

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

def water_to_spout_distance_across_sessions(LogDfs, paths, task_name, animal_id, axes = None):
    """ 
        Plots the evolution of the distance between the spout and water tips across sessions
        Fig.1D Galinanes
    """



    return

def trial_phases_across_sessions(LogDfs, paths, task_name, animal_id, axes = None):
    " Plots the phase of the trial averaged for all trials across sessions Fig.1F Galinanes"

    return axes