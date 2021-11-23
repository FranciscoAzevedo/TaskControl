import scipy as sp
import pandas as pd
import numpy as np
import os 
from pathlib import Path
from Utils import utils
import datetime
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

"""
 
 ########     ###    ########   ######  ######## ########  
 ##     ##   ## ##   ##     ## ##    ## ##       ##     ## 
 ##     ##  ##   ##  ##     ## ##       ##       ##     ## 
 ########  ##     ## ########   ######  ######   ########  
 ##        ######### ##   ##         ## ##       ##   ##   
 ##        ##     ## ##    ##  ##    ## ##       ##    ##  
 ##        ##     ## ##     ##  ######  ######## ##     ## 
 
"""

def get_LogDf_from_path(log_path, return_check=False):
    """ helper to infer task name and get code_map """
    # infer
    task_name = '_'.join(log_path.parent.name.split('_')[2:])
    code_map_path = log_path.parent / task_name / "Arduino" / "src" / "event_codes.h"

    # and read
    CodesDf = utils.parse_code_map(code_map_path)
    code_map = dict(zip(CodesDf['code'], CodesDf['name']))


    LogDf = parse_arduino_log(log_path, code_map, return_check=return_check)
    # try:
    #     LogDf = parse_arduino_log(log_path, code_map)
    # except ValueError:
    #     # Dealing with the earlier LogDfs not having X_tresh/Current_zone etc.
    #     LogDf = parse_arduino_log(log_path, code_map, parse_var=False)

    return LogDf

def parse_arduino_log(log_path, code_map=None, parse_var=True, return_check=False):
    """ create a DataFrame representation of an arduino log. If a code map is passed 
    a corresponding decoded column will be created

    for offline use
    """
    with open(log_path, 'r') as fH:
        lines = fH.readlines()
        
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line != '']

    # test for validity
    valid_lines = []
    invalid_lines = []
    for i, line in enumerate(lines):
        if len(line.split('\t')) == 2 or line.startswith('<'):
            valid_lines.append(line)
        else:
            invalid_lines.append(line)
            utils.printer("bad line in log: %i: %s" % (i, line),"error")

    if len(invalid_lines) == 0:
        all_good = True
    else:
        all_good = False

    if return_check == True:
        if all_good == True:
            return parse_lines(valid_lines, code_map=code_map, parse_var=parse_var)
        else:
            return None
    else:
        return parse_lines(valid_lines, code_map=code_map, parse_var=parse_var)

# TODO merge this one with the CAM reader one
def correct_wraparound(Df, col='t'):
    """ tests and corrects for time wraparound on column t """
    from copy import copy
    _Df = copy(Df)
    if np.any(np.diff(Df['t']) < 0):
        reversal_ind = np.where(np.diff(Df['t']) < 0)[0][0]
        Df['t'].iloc[reversal_ind+1:] += _Df['t'].iloc[reversal_ind]
    return Df

def parse_lines(lines, code_map=None, parse_var=False):
    """ parses a list of lines from arduino into a pd.DataFrame """
    LogDf = pd.DataFrame([line.split('\t') for line in lines if '\t' in line], columns=['code', 't'])
    LogDf['t'] = LogDf['t'].astype('float')
    LogDf = correct_wraparound(LogDf)
    LogDf = LogDf.reset_index(drop=True)

    for col in ['name', 'var', 'value']:
        LogDf[col] = np.NaN

    # decode
    if code_map is not None:
        LogDf['name'] = [code_map[code] for code in LogDf['code']]

    if parse_var:
        var_lines = [line.strip() for line in lines if line.startswith('<VAR')]
        VarDf = pd.DataFrame([line[1:-1].split(' ') for line in var_lines], columns=['_', 'var', 'value', 't'])
        VarDf = VarDf.drop('_', axis=1)
        VarDf['t'] = VarDf['t'].astype('float')
        VarDf['value'] = VarDf['value'].astype('float')
        
        VarDf = correct_wraparound(VarDf)
        
        # join
        LogDf = LogDf.append(VarDf, ignore_index=True, sort=True)
        LogDf = LogDf.sort_values('t')
        LogDf = LogDf.reset_index(drop=True)

    return LogDf

"""
 
 ######## ##     ## ######## ##    ## ########  ######  
 ##       ##     ## ##       ###   ##    ##    ##    ## 
 ##       ##     ## ##       ####  ##    ##    ##       
 ######   ##     ## ######   ## ## ##    ##     ######  
 ##        ##   ##  ##       ##  ####    ##          ## 
 ##         ## ##   ##       ##   ###    ##    ##    ## 
 ########    ###    ######## ##    ##    ##     ######  
 
"""

def get_events_from_name(LogDf, event_name):
    """ extracts event times from LogDf as a pd.DataFrame """
    try:
        EventsDf = LogDf.groupby('name').get_group(event_name)[['t']]
    except KeyError:
        # this gets thrown when the event is not in the log
        return pd.DataFrame(columns=['t'])
    return EventsDf

def get_events(LogDf, event_names):
    """ helper for multiple event_names """
    EventsDict = {}
    for event_name in event_names:
        EventsDict[event_name] = get_events_from_name(LogDf, event_name)
 
    return EventsDict

# removal / generalize
def filter_bad_licks(LogDf, min_time=50, max_time=200, remove=False):
    """ 
    Process recorded LICK_ON and LICK_OFF into realistic licks and add them as an event to the LogDf
    TODO generalize to filter event based on duration
    """
    LickSpan = get_spans_from_names(LogDf, 'LICK_ON', 'LICK_OFF')

    bad_licks = np.logical_or(LickSpan['dt'] < min_time , LickSpan['dt'] > max_time)
    LickSpan = LickSpan.loc[~bad_licks]

    # Add lick_event to LogDf
    Lick_Event = pd.DataFrame(np.stack([['NA']*LickSpan.shape[0], LickSpan['t_on'].values, ['LICK_EVENT']*LickSpan.shape[0]]).T, columns=['code', 't', 'name'])
    Lick_Event['t'] = Lick_Event['t'].astype('float')
    LogDf = LogDf.append(Lick_Event)
    LogDf.sort_values('t')

    if remove is True:
        # TODO
        pass

    return LogDf

def add_go_cue_LogDf(LogDf):
    # Add single GO_CUE_EVENT to LogDf
    go_cue_leftDf = get_events_from_name(LogDf, 'GO_CUE_LEFT_EVENT')
    go_cue_rightDf = get_events_from_name(LogDf, 'GO_CUE_RIGHT_EVENT')
    go_cue_Df = pd.merge(go_cue_leftDf, go_cue_rightDf, how = 'outer')

    go_cue_event = pd.DataFrame(np.stack([['NA']*go_cue_Df.shape[0], go_cue_Df['t'].values, ['GO_CUE_EVENT']*go_cue_Df.shape[0]]).T, columns=['code', 't', 'name'])
    go_cue_event['t'] = go_cue_event['t'].astype('float')
    LogDf = LogDf.append(go_cue_event)

    # Reorder Df according to time and reset indexes
    LogDf = LogDf.sort_values('t')
    LogDf = LogDf.reset_index(drop=True)

    return LogDf

def filter_spans_to_event(LogDf, on_name, off_name, t_min=50, t_max=200, name=None):
    """ 
    creates an Event in the LogDf based on min / max duration of Spans
    """
    Spans = get_spans_from_names(LogDf, on_name, off_name)
    bad_inds = np.logical_or(Spans['dt'] < t_min , Spans['dt'] > t_max)
    Spans = Spans.loc[~bad_inds]

    if name is None:
        name = on_name.split('_ON')[0] + '_EVENT'

    # Add event to LogDf
    Event = pd.DataFrame(np.stack([['NA']*Spans.shape[0], Spans['t_on'].values, [name]*Spans.shape[0]]).T, columns=['code', 't', 'name'])
    Event['t'] = Event['t'].astype('float')
    LogDf = LogDf.append(Event)
    LogDf.sort_values('t')

    return LogDf

"""
 
  ######  ########     ###    ##    ##  ######  
 ##    ## ##     ##   ## ##   ###   ## ##    ## 
 ##       ##     ##  ##   ##  ####  ## ##       
  ######  ########  ##     ## ## ## ##  ######  
       ## ##        ######### ##  ####       ## 
 ##    ## ##        ##     ## ##   ### ##    ## 
  ######  ##        ##     ## ##    ##  ######  
 
"""

def get_spans_from_names(LogDf, on_name, off_name):
    """
    like log2span although with arbitrary events
    this function takes care of above problems actually
    """
    try:
        ons = LogDf.groupby('name').get_group(on_name)
        offs = LogDf.groupby('name').get_group(off_name)
    except KeyError:
        # thrown when name not in log - return empty Df
        return pd.DataFrame(columns=['t_on', 't_off', 'dt'])

    ts = []
    for i, tup in enumerate(ons.itertuples()):
        t_on = tup.t
        binds = offs['t'] > t_on
        if np.any(binds.values):
            t_off = offs.iloc[np.argmax(binds.values)]['t']
            ts.append((t_on, t_off))

    SpansDf = pd.DataFrame(ts, columns=['t_on', 't_off'])
    SpansDf['dt'] = SpansDf['t_off'] - SpansDf['t_on']
  
    return SpansDf

def get_spans(LogDf, span_names):
    """ helper to get spans for multple names from LogDf span names
    returns a dict """
    SpansDict = {}
    for span_name in span_names:
        on_name = span_name + '_ON'
        off_name = span_name + '_OFF'
        SpansDict[span_name] = get_spans_from_names(LogDf, on_name, off_name)
    
    return SpansDict

"""
 
 ######## ########  ####    ###    ##        ######  
    ##    ##     ##  ##    ## ##   ##       ##    ## 
    ##    ##     ##  ##   ##   ##  ##       ##       
    ##    ########   ##  ##     ## ##        ######  
    ##    ##   ##    ##  ######### ##             ## 
    ##    ##    ##   ##  ##     ## ##       ##    ## 
    ##    ##     ## #### ##     ## ########  ######  
 
"""

def parse_trial(TrialDf, Metrics):
    """
    TrialDf is a time slice of a LogDf
    Metrics is list of callables that take a TrialDf as their argument
    and returning a Series
    
    returns a one row DataFrame
    notes: does not know about trial number
    """

    if TrialDf.shape[0] == 0:
        return None
    
    else:
        t = TrialDf.iloc[0]['t']

        # getting metrics
        metrics = [Metric(TrialDf) for Metric in Metrics]
        TrialMetricsDf = pd.DataFrame(metrics).T
        
        # correcting dtype
        for metric in metrics:
            TrialMetricsDf[metric.name] = TrialMetricsDf[metric.name].astype(metric.dtype)
        
        # adding time
        TrialMetricsDf['t'] = t

        return TrialMetricsDf
    
def parse_trials(TrialDfs, Metrics):
    """ helper to run parse_trial on multiple trials """
    SessionDf = pd.concat([parse_trial(Df, Metrics) for Df in TrialDfs], axis=0)
    SessionDf = SessionDf.reset_index(drop=True)
  
    return SessionDf
  
"""
 
  ######  ########  ######   ######  ####  #######  ##    ##  ######  
 ##    ## ##       ##    ## ##    ##  ##  ##     ## ###   ## ##    ## 
 ##       ##       ##       ##        ##  ##     ## ####  ## ##       
  ######  ######    ######   ######   ##  ##     ## ## ## ##  ######  
       ## ##             ##       ##  ##  ##     ## ##  ####       ## 
 ##    ## ##       ##    ## ##    ##  ##  ##     ## ##   ### ##    ## 
  ######  ########  ######   ######  ####  #######  ##    ##  ######  
 
"""

def get_SessionDf(LogDf, metrics, trial_entry_event="TRIAL_AVAILABLE_STATE", trial_exit_event="ITI_STATE"):

    TrialSpans = get_spans_from_names(LogDf, trial_entry_event, trial_exit_event)

    TrialDfs = []
    for i, row in tqdm(TrialSpans.iterrows()):
        TrialDfs.append(time_slice(LogDf, row['t_on'], row['t_off']))
    
    SessionDf = parse_trials(TrialDfs, metrics)
    return SessionDf, TrialDfs

def parse_session(SessionDf, Metrics):
    """ Applies 2nd level metrics to a session """

    # Session is input to Metrics - list of callable functions, each a "Metric"
    metrics = [Metric(SessionDf) for Metric in Metrics]
    SessionMetricsDf = pd.DataFrame(metrics).T

    # correcting dtype
    for metric in metrics:
        SessionMetricsDf[metric.name] = SessionMetricsDf[metric.name].astype(metric.dtype)

    return SessionMetricsDf

def parse_sessions(SessionDfs, Metrics):
    """ helper to run parse_session on multiple sessions.
    SessionDfs is a list of SessionDf """

    PerformanceDf = pd.concat([parse_session(SessionDf, Metrics) for SessionDf in SessionDfs])
    PerformanceDf = PerformanceDf.reset_index(drop = 'True')

    return PerformanceDf
    
        

"""
 
  ######  ##       ####  ######  ######## 
 ##    ## ##        ##  ##    ## ##       
 ##       ##        ##  ##       ##       
  ######  ##        ##  ##       ######   
       ## ##        ##  ##       ##       
 ##    ## ##        ##  ##    ## ##       
  ######  ######## ####  ######  ######## 
 
"""

def time_slice(Df, t_min, t_max, col='t', reset_index=True, mode='inclusive'):
    """ helper to slice a dataframe along time (defined by col) """
    vals = Df[col].values
    if mode == 'exclusive':
        binds = np.logical_and(vals > t_min, vals < t_max)
    if mode is 'inclusive':
        binds = np.logical_and(vals >= t_min, vals <= t_max)

    if reset_index:
        Df = Df.reset_index(drop=True)

    return Df.loc[binds]

def event_slice(Df, event_a, event_b, col='name', reset_index=True):
    """ helper function that slices Df along column name from event_a to event_b """
    try:
        ix_start = Df[Df[col] == event_a].index[0]
        ix_stop = Df[Df[col] == event_b].index[0]
    except IndexError:
        # if either of the events are not present, return an empty dataframe
        return pd.DataFrame([],columns=Df.columns) 

    Df = Df.loc[ix_start:ix_stop]

    if reset_index:
        Df = Df.reset_index(drop=True)

    return Df

def event_based_time_slice(Df, event, pre, post, col='name', on='t', Df_to_slice=None):
    """ slice around and event """
    
    if Df_to_slice is None:
        Df_to_slice = Df

    Dfs = []
    times = Df.groupby(col).get_group(event)[on].values
    for t in times:
        Dfs.append(time_slice(Df_to_slice, t+pre, t+post))
    return Dfs

    # Dfs = []
    # times = Df.groupby(col).get_group(event)[on].values
    # for t in times:
    #     Dfs.append(time_slice(Df, t+pre, t+post))
    # return Dfs

def groupby_dict(Df, Dict):
    """ will turn obsolete ... """
    return Df.groupby(list(Dict.keys())).get_group(tuple(Dict.values()))

def intersect(Df, **kwargs):
    """ helper to slice pd.DataFrame, keys select columns, values select rows at columns
    full intersection, returns empty dataframe on KeyError.
    
    This should replace groupby_dict everywhere """
    try:
        if len(kwargs) == 1:
            return Df.groupby(list(kwargs.keys())[0]).get_group(tuple(kwargs.values())[0])
        else:
            try:
                return Df.groupby(list(kwargs.keys())).get_group(tuple(kwargs.values()))
            except IndexError:
                # thrown when more then 1 key are empty
                return pd.DataFrame([],columns=Df.columns)
    except KeyError:
        # thrown when key is not present
        return pd.DataFrame([],columns=Df.columns)

def expand_columns(Df, categorial_cols):
    """ turns a single categorial column into boolean columns with is_ prefix """
    for category_col in categorial_cols:
        categories = Df[category_col].unique()
        categories = [cat for cat in categories if not pd.isna(cat)]
        for category in categories:
            Df['is_'+category] = Df[category_col] == category
    return Df

"""

 ##     ##    ###    ########  ########
 ##     ##   ## ##   ##     ## ##     ##
 ##     ##  ##   ##  ##     ## ##     ##
 ######### ##     ## ########  ########
 ##     ## ######### ##   ##   ##
 ##     ## ##     ## ##    ##  ##
 ##     ## ##     ## ##     ## ##

"""

# SPLIT / REMOVE
def parse_bonsai_LoadCellData(csv_path):
    LoadCellDf = pd.read_csv(csv_path, names=['t','x','y'])
    return LoadCellDf

# def parse_bonsai_LoadCellData(csv_path, save=True, trig_len=1, ttol=0.2):
#     LoadCellDf = pd.read_csv(csv_path, names=['t','x','y'])

#     harp_sync = pd.read_csv(csv_path.parent / "bonsai_harp_sync.csv", names=['t']).values.flatten()
#     t_sync_high = harp_sync[::2]
#     t_sync_low = harp_sync[1::2]

#     dts = np.array(t_sync_low) - np.array(t_sync_high)
#     good_timestamps = ~(np.absolute(dts-trig_len)>ttol)
#     t_sync = np.array(t_sync_high)[good_timestamps]

#     t_sync = pd.DataFrame(t_sync, columns=['t'])
#     if save:
#         # write to disk
#         # LoadCellDf.to_csv(harp_csv_path.parent / "loadcell_data.csv") # obsolete now
#         t_sync.to_csv(csv_path.parent / "harp_sync.csv")

#     return LoadCellDf, t_sync

# obsolete but keep
def parse_harp_csv(harp_csv_path, save=True, trig_len=1, ttol=0.2):
    """ gets the loadcell data and the sync times from a harp csv log
    trig_len is time in ms of sync trig high, tol is deviation in ms
    check harp sampling time, seems to be 10 khz? """

    with open(harp_csv_path, 'r') as fH:
        lines = fH.readlines()

    header = lines[0].split(',')

    t_sync_high = []
    t_sync_low = []
    LoadCellDf = []
    
    for line in tqdm(lines[1:], desc="parsing harp log", position=0, leave=True):
        elements = line.split(',')
        if elements[0] == '3': # line is an event
            if elements[1] == '33': # line is a load cell read
                data = line.split(',')[2:5]
                LoadCellDf.append(data)
            if elements[1] == '34': # line is a digital input timestamp
                line = line.strip().split(',')
                if line[3] == '1': # high values
                    t_sync_high.append(float(line[2])*1000) # convert to ms
                if line[3] == '0': # low values
                    t_sync_low.append(float(line[2])*1000) # convert to ms

    dts = np.array(t_sync_low) - np.array(t_sync_high)
    good_timestamps = ~(np.absolute(dts-trig_len)>ttol)
    t_sync = np.array(t_sync_high)[good_timestamps]

    LoadCellDf = pd.DataFrame(LoadCellDf, columns=['t', 'x', 'y'], dtype='float')
    LoadCellDf['t_original'] = LoadCellDf['t'] # keeps the original
    LoadCellDf['t'] = LoadCellDf['t'] * 1000

    t_sync = pd.DataFrame(t_sync, columns=['t'])
    if save:
        # write to disk
        LoadCellDf.to_csv(harp_csv_path.parent / "loadcell_data.csv")
        t_sync.to_csv(harp_csv_path.parent / "harp_sync.csv")
    
    return LoadCellDf, t_sync

"""
 ######  ########    ###    ########  ######
##    ##    ##      ## ##      ##    ##    ##
##          ##     ##   ##     ##    ##
 ######     ##    ##     ##    ##     ######
      ##    ##    #########    ##          ##
##    ##    ##    ##     ##    ##    ##    ##
 ######     ##    ##     ##    ##     ######
"""
from sklearn.linear_model import LogisticRegression
from scipy.special import expit


def log_reg_sklearn(x, y, x_fit=None):
    """ x and y are of shape (N, ) y are choices in [0, 1] """
    if x_fit is None:
        x_fit = np.linspace(x.min(), x.max(), 100)

    cLR = LogisticRegression()
    try:
        cLR.fit(x[:, np.newaxis], y)
        y_fit = expit(x_fit * cLR.coef_ + cLR.intercept_).flatten()
    except ValueError:
        y_fit = sp.zeros(x_fit.shape)
        y_fit[:] = sp.nan

    return y_fit, (cLR.coef_, cLR.intercept_)

def psychometric(x, x0, k):
    return 1 / (1+np.exp(-k * (x-x0)))

def psychometric_w_lapses(x, x0, k, Lu, Ll):
    return Lu / (1+np.exp(-k * (x-x0))) + Ll

def log_reg_cf(x, y, x_fit=None, fit_lapses=True):
    """ x and y are of shape (N, ) y are choices in [0, 1] """
    from scipy.optimize import curve_fit

    if x_fit is None:
        x_fit = np.linspace(x.min(), x.max(), 100)

    if fit_lapses:
        fun = psychometric_w_lapses
        bounds = ((0,3000), (-0.1, 0.1), (0,1), (0,1))
        p0 = (1500, 0.0, 0, 1)
    else:
        fun = psychometric
        bounds = ((0, 3000), (-0.1, 0.1))
        p0 = (1500, 0)

    pfit = curve_fit(fun, x, y, p0, bounds=np.array(bounds).T)[0]
    y_fit = fun(x_fit, *pfit)
    return y_fit, pfit

def log_reg(x, y, x_fit=None, fit_lapses=True):
    """ x and y are of shape (N, ) y are choices in [0, 1] """

    """ note - this is broken """
    
    from scipy.optimize import minimize

    if x_fit is None:
        x_fit = np.linspace(x.min(), x.max(), 100)

    def obj_fun(p, x, y, fun):
        yhat = fun(x, *p)
        Rss = np.sum((y-yhat)**2)
        return Rss

    if fit_lapses:
        fun = psychometric_w_lapses
        bounds = ((0,3000), (-0.1, 0.1), (0,1), (0,1))
        p0 = (1500, 0.005, 0, 1)
    else:
        fun = psychometric
        bounds = ((0, 3000), (-0.1, 0.1))
        p0 = (1500, 0.005)

    pfit = minimize(obj_fun, p0, args=(x, y, fun))
    y_fit = fun(x_fit, *pfit.x)
    return y_fit, pfit.x

def tolerant_mean(arrs):
    'A mean that is tolerant to different sized arrays'

    max_length = np.max([arr.shape[0] for arr in arrs]) # get largest array

    # suggestion
    # A = np.zeros((len(arrs),max_length))
    # A[:] = np.NaN
    # for i, arr in enumerate(arrs):
    #     A[:arr.shape[0],i] = arr
    # return np.nanmean(A,axis=0)

    arrs=[np.pad(arr, (0, max_length-arr.shape[0]), mode='constant', constant_values=np.nan) for arr in arrs] # pad every array until max_length to obtain square matrix

    return np.nanmean(arrs, axis = 0)
