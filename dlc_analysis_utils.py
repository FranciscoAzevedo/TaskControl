import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipy as sp
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from copy import copy

# Custom
import behavior_analysis_utils as bhv
from metrics import *

"""
 
 ########  ########    ###    ########  
 ##     ## ##         ## ##   ##     ## 
 ##     ## ##        ##   ##  ##     ## 
 ########  ######   ##     ## ##     ## 
 ##   ##   ##       ######### ##     ## 
 ##    ##  ##       ##     ## ##     ## 
 ##     ## ######## ##     ## ########  
 
"""

def read_dlc_csv(csv_path):
    """ reads Dlc csv output to hierarchical multiindex (on columns) pd.DataFrame """
    DlcDf = pd.read_csv(csv_path, header=[1,2],index_col=0)
    return DlcDf

def read_dlc_h5(path):
    """ much faster than read_dlc_csv() """
    DlcDf = pd.read_hdf(path)
    DlcDf = DlcDf[DlcDf.columns.levels[0][0]]
    return DlcDf

def read_video(path):
    Vid = cv2.VideoCapture(str(path))
    return Vid

"""
 
  ######  ##    ## ##    ##  ######  
 ##    ##  ##  ##  ###   ## ##    ## 
 ##         ####   ####  ## ##       
  ######     ##    ## ## ## ##       
       ##    ##    ##  #### ##       
 ##    ##    ##    ##   ### ##    ## 
  ######     ##    ##    ##  ######  
 
"""

def sync_arduino_w_dlc(log_path, video_sync_path):
    Arduino_SyncEvent = bhv.get_arduino_sync(log_path)

    SyncDf = pd.read_csv(video_sync_path,names=['frame','t','GPIO'])

    # dealing with the the multiple wraparounds of the cam clock
    
    SyncDf_orig = copy(SyncDf)

    while np.any(np.diff(SyncDf['t']) < 0):
        reversal_ind = np.where(np.diff(SyncDf['t']) < 0)[0][0]
        SyncDf['t'].iloc[reversal_ind+1:] += SyncDf_orig['t'].iloc[reversal_ind]

    ons = sp.where(sp.diff(SyncDf.GPIO) > 1)[0]
    # offs = sp.where(sp.diff(SyncDf.GPIO) < -1)[0] # can be used to check correct length

    Camera_SyncEvent = SyncDf.iloc[ons+1] # one frame offset
    
    # check for unequal
    if Arduino_SyncEvent.shape[0] != Camera_SyncEvent.shape[0]:
        print('unequal sync pulses: Arduino: %i, Camera: %i' % (Arduino_SyncEvent.shape[0],Camera_SyncEvent.shape[0]))
        t_arduino, t_camera, offset = bhv.cut_timestamps(Arduino_SyncEvent.t.values,Camera_SyncEvent.t.values,verbose=True, return_offset=True)
        frames_index = Camera_SyncEvent.index.values[offset:offset+t_arduino.shape[0]]
    else:
        t_arduino = Arduino_SyncEvent.t.values
        t_camera = Camera_SyncEvent.t.values
        frames_index = Camera_SyncEvent.index.values
        
    
    # linear regressions linking arduino times to frames and vice versa
    from scipy import stats
    
    # from arduino time to camera time
    m, b = stats.linregress(t_camera, t_arduino)[:2]

    # from camera time to camera frame
    m2, b2 = stats.linregress(frames_index, t_camera)[:2]

#    # from arduino time to camera time
#    m, b = stats.linregress(Arduino_SyncEvent.t.values, Camera_SyncEvent.t.values)[:2]
#
#    # from camera time to camera frame
#    m2, b2 = stats.linregress(Camera_SyncEvent.t.values, Camera_SyncEvent.index.values)[:2]

    return m, b, m2, b2

def time2frame(i,m,b,m2,b2):
    return (((i-b2)/m2)-b)/m

def  frame2time(t,m,b,m2,b2):
    return sp.int32((t*m+b)*m2+b2)

"""
 
 ########  ##        #######  ######## 
 ##     ## ##       ##     ##    ##    
 ##     ## ##       ##     ##    ##    
 ########  ##       ##     ##    ##    
 ##        ##       ##     ##    ##    
 ##        ##       ##     ##    ##    
 ##        ########  #######     ##    
 
"""

def get_frame(Vid, i):
    """ Vid is cv2 VideoCapture obj """
    Vid.set(1,i)
    Frame = Vid.read()[1][:,:,0] # fetching r, ignoring g b, all the same
    # TODO check if monochrome can be specified in VideoCaputre
    return Frame

def plot_frame(Frame, axes=None, **im_kwargs):
    if axes is None:
        fig, axes = plt.subplots()
        axes.set_aspect('equal')
    
    defaults  = dict(cmap='gray')
    for k,v in defaults.items():
        im_kwargs.setdefault(k,v)

    axes.imshow(Frame, **im_kwargs)
    return axes

def plot_bodyparts(bodyparts, DlcDf, i , axes=None, **marker_kwargs):
    if axes is None:
        fig, axes = plt.subplots()

    df = DlcDf.loc[i]
    for bp in bodyparts:
        axes.plot(df[bp].x,df[bp].y,'o', **marker_kwargs)

    return axes

def plot_Skeleton(Skeleton, DlcDf, i, axes=None,**line_kwargs):
    if axes is None:
        fig, axes = plt.subplots()

    defaults  = dict(lw=1,alpha=0.5,color='k')
    for k,v in defaults.items():
        line_kwargs.setdefault(k,v)

    df = DlcDf.loc[i]

    lines = []
    for node in Skeleton:
        line, = axes.plot([df[node[0]].x,df[node[1]].x], [df[node[0]].y,df[node[1]].y], **line_kwargs)
        lines.append(line)

    return axes, lines

def plot_trajectories(DlcDf, bodyparts, axes=None, p=0.99, **line_kwargs):
    if axes is None:
        fig, axes = plt.subplots()
        axes.set_aspect('equal')
    
    defaults  = dict(lw=0.05, alpha=0.85)
    for k,v in defaults.items():
        line_kwargs.setdefault(k,v)

    for bp in bodyparts:
        df = DlcDf[bp]
        ix = df.likelihood > p
        df = df.loc[ix]
        axes.plot(df.x, df.y, **line_kwargs)

    return axes


"""
########  ##        #######  ########    ##     ## ######## ##       ########  ######## ########   ######
##     ## ##       ##     ##    ##       ##     ## ##       ##       ##     ## ##       ##     ## ##    ##
##     ## ##       ##     ##    ##       ##     ## ##       ##       ##     ## ##       ##     ## ##
########  ##       ##     ##    ##       ######### ######   ##       ########  ######   ########   ######
##        ##       ##     ##    ##       ##     ## ##       ##       ##        ##       ##   ##         ##
##        ##       ##     ##    ##       ##     ## ##       ##       ##        ##       ##    ##  ##    ##
##        ########  #######     ##       ##     ## ######## ######## ##        ######## ##     ##  ######
"""

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

def filter_trials_by(SessionDf, TrialDfs, filter_pairs):
    """
        This function filters input TrialDfs given filter_pair tuple (or list of tuples)
        Example: given filter_pairs [(outcome,correct) , (choice,left)] it will only output trials which are correct to left side 
    """

    if type(filter_pairs) is list: # in case its more than one pair
        groupby_keys = [filter_pair[0] for filter_pair in filter_pairs]
        getgroup_keys = tuple([filter_pair[1] for filter_pair in filter_pairs])
    else:
        groupby_keys = filter_pairs[0]
        getgroup_keys = filter_pairs[1]

    try:
        SDf = SessionDf.groupby(groupby_keys).get_group(getgroup_keys)
    except:
        print('There are no trials with given input filter_pair combination')
        raise KeyError

    TrialDfs_filt = np.array(TrialDfs, dtype="object")[SDf.index.values.astype(int)]

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
    """
        Returns Fx/Fy/Fmag NUMPY ND.ARRAY for all trials aligned to an event in a window defined by [align-pre, align+post]"
        Don't forget: we are using nd.arrays so, implicitly, we use advanced slicing which means [row, col] instead of [row][col]
    """
    X, Y = [],[]

    for TrialDf in TrialDfs:
        
        t_align = TrialDf.loc[TrialDf['name'] == align_event, 't'].values[0]
        LCDf = bhv.time_slice(LoadCellDf, t_align-pre, t_align+post) # slice around reference event

        # Store
        X.append(LCDf['x'].values)
        Y.append(LCDf['y'].values)

    # Turn into numpy arrays
    X = np.array(X).T
    Y = np.array(Y).T

    return X,Y

"""
 
    ###    ##    ##    ###    ##       ##    ##  ######  ####  ######  
   ## ##   ###   ##   ## ##   ##        ##  ##  ##    ##  ##  ##    ## 
  ##   ##  ####  ##  ##   ##  ##         ####   ##        ##  ##       
 ##     ## ## ## ## ##     ## ##          ##     ######   ##   ######  
 ######### ##  #### ######### ##          ##          ##  ##        ## 
 ##     ## ##   ### ##     ## ##          ##    ##    ##  ##  ##    ## 
 ##     ## ##    ## ##     ## ########    ##     ######  ####  ######  
 
"""

def add_go_cue_LogDf(LogDf):
    # Add single GO_CUE_EVENT to LogDf
    go_cue_leftDf = bhv.get_events_from_name(LogDf, 'GO_CUE_LEFT_EVENT')
    go_cue_rightDf = bhv.get_events_from_name(LogDf, 'GO_CUE_RIGHT_EVENT')
    go_cue_Df = pd.merge(go_cue_leftDf, go_cue_rightDf, how = 'outer')

    go_cue_event = pd.DataFrame(np.stack([['NA']*go_cue_Df.shape[0], go_cue_Df['t'].values, ['GO_CUE_EVENT']*go_cue_Df.shape[0]]).T, columns=['code', 't', 'name'])
    go_cue_event['t'] = go_cue_event['t'].astype('float')
    LogDf = LogDf.append(go_cue_event)

    # Reorder Df according to time and reset indexes
    LogDf = LogDf.sort_values('t')
    LogDf = LogDf.reset_index(drop=True)

    return LogDf

def box2rect(center, w):
    """ definition: x1,y1, x2,y2 """
    w2 = int(w/2)
    return (center[0]-w2, center[1]-w2, center[0]+w2, center[1]+w2)

def rect2cart(rect):
    """ helper for matplotlib """
    xy = (rect[0],rect[1])
    width = rect[2]-rect[0]
    height = rect[3]-rect[1]
    return xy, width, height

def get_in_box(DlcDf, bp, rect, p=0.99, filter=False):
    """ returns boolean vector over frames if bodypart is in
    rect and returns boolean index for above pred above likelihood """

    df = DlcDf[bp]
    x_true = sp.logical_and((df.x > rect[0]).values, (df.x < rect[2]).values)
    y_true = sp.logical_and((df.y > rect[1]).values, (df.y < rect[3]).values)
    in_box = sp.logical_and(x_true, y_true)
    good_ix = (df.likelihood > p).values
    if filter is False:
        return in_box, good_ix
    else:
        in_box[~good_ix] = False
        return in_box    

def in_box_span(DlcDf, bp, rect, p=0.99, min_dur=20, convert_to_time=True):
    """ returns a SpansDf for body part in box times """

    in_box = get_in_box(DlcDf, bp, rect, p=p, filter=True)

    df = pd.DataFrame(columns=['t_on','t_off'])

    ons = np.where(np.diff(in_box.astype('int32')) == 1)[0]
    offs = np.where(np.diff(in_box.astype('int32')) == -1)[0]

    ts = []
    for t_on in ons:
        binds = offs > t_on
        if np.any(binds):
            t_off = offs[np.argmax(binds)]
            ts.append((t_on,t_off))

    SpansDf = pd.DataFrame(ts, columns=['t_on', 't_off'])
    SpansDf['dt'] = SpansDf['t_off'] - SpansDf['t_on']

    # filter min dur
    SpansDf = SpansDf[SpansDf.dt > min_dur]

    # if convert_to_time:
    #     SpansDf = pd.DataFrame(frame2time(SpansDf.values,m,b,m2,b2),columns=SpansDf.columns)

    return SpansDf

"""
 
 ##     ## ######## ######## ########  ####  ######   ######  
 ###   ### ##          ##    ##     ##  ##  ##    ## ##    ## 
 #### #### ##          ##    ##     ##  ##  ##       ##       
 ## ### ## ######      ##    ########   ##  ##        ######  
 ##     ## ##          ##    ##   ##    ##  ##             ## 
 ##     ## ##          ##    ##    ##   ##  ##    ## ##    ## 
 ##     ## ########    ##    ##     ## ####  ######   ######  
 
"""

def calc_grasp_phase(DlcDf, bps, p=0.99, filter=False):
    " Relative phase of grasp across trial - TODO , stub for now"
    p_phase = []

    return p_phase

def calc_dist_bp_point(DlcDf, bp, point, p=0.99, filter=False):
    """ euclidean distance bodypart to point """

    df = DlcDf[bp]
    d = sp.sqrt(sp.sum((df[['x','y']].values - sp.array(point))**2,axis=1))
    good_ix = (df.likelihood > p).values
    if filter is False:
        return d, good_ix
    else:
        d[~good_ix] = sp.nan
        return d

def calc_dist_bp_bp(DlcDf, bp1, bp2, p=0.99, filter=False):
    """ euclidean distance between bodyparts """

    df1 = DlcDf[bp1]
    df2 = DlcDf[bp2]

    c1 = df1[['x','y']].values
    c2 = df2[['x','y']].values

    good_ix = sp.logical_and((df1.likelihood > p).values,(df2.likelihood > p).values)

    d = sp.sqrt(sp.sum((c1-c2)**2,axis=1))
    if filter is False:
        return d, good_ix
    else:
        d[~good_ix] = sp.nan
        return d

def get_speed(DlcDf, bp, p=0.99, filter=False):
    """ bodypart speed over time in px/ms """

    Vxy = sp.diff(DlcDf[bp][['x','y']].values,axis=0) / DlcDf['t'][:-1].values[:,sp.newaxis]
    V = sp.sqrt(sp.sum(Vxy**2,axis=1)) # euclid vector norm
    V = V / sp.diff(DlcDf['t'].values) # -> to speed

    V = sp.concatenate([[sp.nan],V]) # pad first to nan (speed undefined)
    good_ix = (DlcDf[bp].likelihood > p).values

    if filter is False:
        return V, good_ix
    else:
        V[~good_ix] = sp.nan
        return V

# Work only on Trial-level
def has_reach_left(TrialDf):
    var_name = 'has_reach_left'

    if "REACH_LEFT_ON" in TrialDf['name'].values:
        var = True
    else:
        var = False    
 
    return pd.Series(var, name=var_name)

def has_reach_right(TrialDf):
    var_name = 'has_reach_right'

    if "REACH_RIGHT_ON" in TrialDf['name'].values:
        var = True
    else:
        var = False    
 
    return pd.Series(var, name=var_name)

def choice_rt_left(TrialDf):
    var_name = 'choice_rt_left'

    if get_chosen_side(TrialDf).values == 'left':
        var = get_choice_rt(TrialDf)
    else:
        var = np.NaN

    return pd.Series(var, name=var_name)

def choice_rt_right(TrialDf):
    var_name = 'choice_rt_right'

    if get_chosen_side(TrialDf).values == 'right':
        var = get_choice_rt(TrialDf)
    else:
        var = np.NaN

    return pd.Series(var, name=var_name)

def get_delay_rt(TrialDf):
    " First reach RT during only the DELAY period, agnostic of chosen side or arm"
    var_name = 'delay_rt'

    Df = event_slice(TrialDf, "PRESENT_INTERVAL_STATE", "GO_CUE_EVENT")

    # Union of left and right reaches
    reach_leftDf = bhv.get_events_from_name(Df, 'REACH_LEFT_ON')
    reach_rightDf = bhv.get_events_from_name(Df, 'REACH_RIGHT_ON')
    reach_Df  =pd.merge(reach_leftDf,reach_rightDf, how = 'outer')

    # compute the time of the delay rt
    if not reach_Df.empty:
        var = reach_Df.iloc[0]['t'] - Df.iloc[0]['t']
    else:
        var = np.NaN
        
    return pd.Series(var, name=var_name)

# Backwards compatible for v0.2 (didnt have CHOICE_EVENT)
def choice_rt_left_b(TrialDf):
    var_name = 'choice_rt_left'
    
    if has_reach_left(TrialDf).values:

        # for learn to reach and learn to choose
        if 'PRESENT_CUE_STATE' in TrialDf['name'].values: 
            cue_time = TrialDf.groupby('name').get_group("PRESENT_CUE_STATE").iloc[-1]['t']

        # for learn to init, fixate and time 
        if get_outcome(TrialDf).values == 'premature': # in case it is a premature choice
            cue_time = TrialDf.groupby('name').get_group("PREMATURE_CHOICE_EVENT").iloc[-1]['t']
        elif 'GO_CUE_LEFT_EVENT' in TrialDf['name'].values: # go cue on left
            cue_time = TrialDf.groupby('name').get_group("GO_CUE_LEFT_EVENT").iloc[-1]['t']
        elif 'GO_CUE_RIGHT_EVENT' in TrialDf['name'].values: # go cue on right
            cue_time =  TrialDf.groupby('name').get_group("GO_CUE_RIGHT_EVENT").iloc[-1]['t']

        # only first reach
        reach_times = TrialDf.groupby('name').get_group("REACH_LEFT_ON")['t'] 
        rt = reach_times[reach_times>cue_time].values

        if len(rt) != 0:
            var = rt[0] - cue_time 
        else:
            var = np.NaN
    else:
        var = np.NaN    
 
    return pd.Series(var, name=var_name)

def choice_rt_right_b(TrialDf):
    var_name = 'choice_rt_right'

    if has_reach_right(TrialDf).values:

        # for learn to reach and learn to choose
        if 'PRESENT_CUE_STATE' in TrialDf['name'].values: 
            cue_time = TrialDf.groupby('name').get_group("PRESENT_CUE_STATE").iloc[-1]['t']

        # for learn to init, fixate and time 
        if get_outcome(TrialDf).values == 'premature': # in case it is a premature choice
            cue_time = TrialDf.groupby('name').get_group("PREMATURE_CHOICE_EVENT").iloc[-1]['t']
        elif 'GO_CUE_LEFT_EVENT' in TrialDf['name'].values: # go cue on left
            cue_time = TrialDf.groupby('name').get_group("GO_CUE_LEFT_EVENT").iloc[-1]['t']
        elif 'GO_CUE_RIGHT_EVENT' in TrialDf['name'].values: # go cue on right
            cue_time = TrialDf.groupby('name').get_group("GO_CUE_RIGHT_EVENT").iloc[-1]['t']

        # only first reach
        reach_times = TrialDf.groupby('name').get_group("REACH_RIGHT_ON")['t']
        rt = reach_times[reach_times>cue_time].values

        if len(rt) != 0:
            var = rt[0] - cue_time 
        else:
            var = np.NaN
    else:
        var = np.NaN    
 
    return pd.Series(var, name=var_name)