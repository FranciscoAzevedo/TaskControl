import pandas as pd
import numpy as np
from Utils import behavior_analysis_utils as bhv

"""
 
  ######  ##     ##  #######  ####  ######  ######## 
 ##    ## ##     ## ##     ##  ##  ##    ## ##       
 ##       ##     ## ##     ##  ##  ##       ##       
 ##       ######### ##     ##  ##  ##       ######   
 ##       ##     ## ##     ##  ##  ##       ##       
 ##    ## ##     ## ##     ##  ##  ##    ## ##       
  ######  ##     ##  #######  ####  ######  ######## 
 
"""
def has_choice(TrialDf):
    var_name = 'has_choice'

    if "CHOICE_EVENT" in TrialDf['name'].values:
        var = True
    else:
        var = False    
 
    return pd.Series(var, name=var_name)

def get_chosen_side(TrialDf):
    var_name = "chosen_side"

    if "CHOICE_EVENT" in TrialDf['name'].values:
        if "CHOICE_LEFT_EVENT" in TrialDf['name'].values:
            var = "left"
        elif "CHOICE_RIGHT_EVENT" in TrialDf['name'].values:
            var = "right"
    else:
        var = np.NaN

    return pd.Series(var, name=var_name)

def get_chosen_interval(TrialDf):
    var_name = "chosen_interval"

    if "CHOICE_EVENT" in TrialDf['name'].values:
        if "CHOICE_SHORT_EVENT" in TrialDf['name'].values:
            var = "short"
        elif "CHOICE_LONG_EVENT" in TrialDf['name'].values:
            var = "long"
    else:
        var = np.NaN

    return pd.Series(var, name=var_name)


def get_correct_side(TrialDf):
    var_name = "correct_side"
    try:
        Df = TrialDf.groupby('var').get_group(var_name)
        val = Df.iloc[0]['value']
        if val == 4:
            var = "left"
        if val == 6:
            var = "right"

    except KeyError:
        var = np.NaN

    return pd.Series(var, name=var_name)

def get_interval_category(TrialDf):
    var_name = "interval_category"
    try:
        interval = get_interval(TrialDf).values[0]
        if interval < 1500:
            var = "short"
        else:
            var = "long"

    except KeyError:
        var = np.NaN
    
    return pd.Series(var, name=var_name)

def get_interval(TrialDf):
    var_name = "this_interval"
    try:
        Df = TrialDf.groupby('var').get_group(var_name)
        var = Df.iloc[0]['value']
    except KeyError:
        var = np.NaN

    return pd.Series(var, name=var_name)

def get_outcome(TrialDf):
    var_name = "outcome"

    if "CHOICE_MISSED_EVENT" in TrialDf['name'].values:
        var = "missed"
    elif "CHOICE_INCORRECT_EVENT" in TrialDf['name'].values:
        var = "incorrect"
    elif "CHOICE_CORRECT_EVENT" in TrialDf['name'].values:
        var = "correct"
    elif "PREMATURE_CHOICE_EVENT" in TrialDf['name'].values:
        var = "premature"
    else:
        var = np.NaN

    return pd.Series(var, name = var_name)

# Paco
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

def get_reached_side(TrialDf):
    var_name = 'reached_side'

    if has_reach_left(TrialDf)[0] and has_reach_right(TrialDf)[0]:
        var = 'both'
    elif has_reach_left(TrialDf)[0]:
        var = 'left'
    elif has_reach_right(TrialDf)[0]:
        var = 'right'
    else:
        var = np.NaN

    return pd.Series(var, name=var_name)

def reach_rt_left(TrialDf):
    " only makes sense to use if mistakes are allowed "
    var_name = 'reach_rt_left'

    try:
        Df = bhv.event_slice(TrialDf, "CHOICE_STATE", "REACH_LEFT_ON")
        var = Df.iloc[-1]['t'] - Df.iloc[0]['t']
    except:
        var = np.NaN

    return pd.Series(var, name=var_name)

def reach_rt_right(TrialDf):
    " only makes sense to use if mistakes are allowed "
    var_name = 'reach_rt_right'

    try:
        Df = bhv.event_slice(TrialDf, "CHOICE_STATE", "REACH_RIGHT_ON")
        var = Df.iloc[-1]['t'] - Df.iloc[0]['t']
    except:
        var = np.NaN

    return pd.Series(var, name=var_name)

def delay_rt(TrialDf):
    " First reach RT during only the DELAY period, agnostic of chosen side or arm"
    var_name = 'delay_rt'

    Df = bhv.event_slice(TrialDf, "PRESENT_INTERVAL_STATE", "GO_CUE_EVENT")

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

def get_reach_type(DLCDf,TrialDf):
    " Classifies type of reach (contra,ipsi,double) and to each side"
    var_name = 'reach_type'
    var = []

    #if has_reach_left(TrialDf):
        # Check for ipsi vs contra
        

        # Double case

    #if has_reach_right(TrialDf):
        # Check for ipsi vs contra

        # Double case

    return pd.Series(var, name=var_name)


"""
 
 ######## ########  ####    ###    ##          ######## ##    ## ########  ######## 
    ##    ##     ##  ##    ## ##   ##             ##     ##  ##  ##     ## ##       
    ##    ##     ##  ##   ##   ##  ##             ##      ####   ##     ## ##       
    ##    ########   ##  ##     ## ##             ##       ##    ########  ######   
    ##    ##   ##    ##  ######### ##             ##       ##    ##        ##       
    ##    ##    ##   ##  ##     ## ##             ##       ##    ##        ##       
    ##    ##     ## #### ##     ## ########       ##       ##    ##        ######## 
 
"""

def get_in_corr_loop(TrialDf):
    var_name = "in_corr_loop"
    try:
        Df = TrialDf.groupby('var').get_group(var_name)
        var = Df.iloc[0]['value'].astype('bool')
    except KeyError:
        var = np.NaN

    return pd.Series(var, name=var_name)

def get_timing_trial(TrialDf):
    var_name = "timing_trial"
    try:
        Df = TrialDf.groupby('var').get_group(var_name)
        var = Df.iloc[0]['value'].astype('bool')
    except KeyError:
        var = np.NaN

    return pd.Series(var, name=var_name)

"""
 
 ######## #### ##     ## #### ##    ##  ######   
    ##     ##  ###   ###  ##  ###   ## ##    ##  
    ##     ##  #### ####  ##  ####  ## ##        
    ##     ##  ## ### ##  ##  ## ## ## ##   #### 
    ##     ##  ##     ##  ##  ##  #### ##    ##  
    ##     ##  ##     ##  ##  ##   ### ##    ##  
    ##    #### ##     ## #### ##    ##  ######   
 
"""

def get_start(TrialDf):
    return pd.Series(TrialDf.iloc[0]['t'], name='t_on')

def get_stop(TrialDf):
    return pd.Series(TrialDf.iloc[-1]['t'], name='t_off')

def get_init_rt(TrialDf):
    var_name = "init_rt"
    try:
        Df = bhv.event_slice(TrialDf, "TRIAL_AVAILABLE_EVENT", "TRIAL_ENTRY_EVENT")
        var = Df.iloc[-1]['t'] - Df.iloc[0]['t']
    except IndexError:
        var = np.NaN
    return pd.Series(var, name=var_name)

def get_premature_rt(TrialDf):
    var_name = "premature_rt"
    try:
        Df = bhv.event_slice(TrialDf, "PRESENT_INTERVAL_STATE", "PREMATURE_CHOICE_EVENT")
        var = Df.iloc[-1]['t'] - Df.iloc[0]['t']
    except IndexError:
        var = np.NaN
    return pd.Series(var, name=var_name)

def get_choice_rt(TrialDf):
    var_name = "choice_rt"
    try:
        Df = bhv.event_slice(TrialDf, "CHOICE_STATE", "CHOICE_EVENT")
        var = Df.iloc[-1]['t'] - Df.iloc[0]['t']
    except IndexError:
        var = np.NaN
    return pd.Series(var, name=var_name)

