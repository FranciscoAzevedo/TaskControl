import pandas as pd
import numpy as np
from Utils import behavior_analysis_utils as bhv
from Utils import dlc_analysis_utils as dlc_utils
from Utils.behavior_analysis_utils import event_slice
from Utils import utils

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

def has_anticipatory_reach(TrialDf):
    var_name = 'has_anticip_reach'
    if "ANTICIPATORY_REACH_EVENT" in TrialDf['name'].values:
        var = True
    else:
        var = False    
 
    return pd.Series(var, name=var_name)

def has_premature_choice(TrialDf):
    var_name = "has_premature_choice"
    if "PREMATURE_CHOICE_EVENT" in TrialDf['name'].values:
        var = True
    else:
        var = False    
 
    return pd.Series(var, name=var_name)

def has_reward_collected(TrialDf):
    var_name = "has_reward_collected"
    if "REWARD_COLLECTED_EVENT" in TrialDf['name'].values:
        var = True
    else:
        var = False    
 
    return pd.Series(var, name=var_name)

def has_autodelivered_reward(TrialDf):
    var_name = "has_autodelivered_reward"
    if "REWARD_AUTODELIVERED_EVENT" in TrialDf['name'].values:
        var = True
    else:
        var = False    
 
    return pd.Series(var, name=var_name)

def has_premature_reach(TrialDf):
    var_name = "has_premature_reach"
    try:
        Df = event_slice(TrialDf, "PRESENT_INTERVAL_STATE", "CHOICE_STATE")
        events = Df['name'].values
        if "REACH_LEFT_ON" in events or "REACH_RIGHT_ON" in events:
            var = True
        else:
            var = False
    except KeyError:
        # this should take care of incomplete trials
        var = np.NaN

    return pd.Series(var, name=var_name)

def get_chosen_side(TrialDf):
    var_name = "chosen_side"

    if "CHOICE_EVENT" in TrialDf['name'].values:

        choice_leftDf = bhv.get_events_from_name(TrialDf, 'CHOICE_LEFT_EVENT')
        choice_rightDf = bhv.get_events_from_name(TrialDf, 'CHOICE_RIGHT_EVENT')

        if len(choice_leftDf) != 0 and len(choice_rightDf) == 0:
            var = 'left'

        elif len(choice_leftDf) == 0 and len(choice_rightDf) != 0:
            var = 'right'
        
        # Both sides clock choices fast
        elif len(choice_leftDf) != 0 and len(choice_rightDf) != 0:
            if choice_leftDf.iloc[0]['t'] < choice_rightDf.iloc[0]['t']:
                var = 'left'
            else:
                var = 'right'

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

    if "CHOICE_INCORRECT_EVENT" in TrialDf['name'].values:
        var = "incorrect"
    if "PREMATURE_CHOICE_EVENT" in TrialDf['name'].values:
        var = "premature"
    elif "CHOICE_CORRECT_EVENT" in TrialDf['name'].values:
        var = "correct"
    elif "ANTICIPATORY_REACH_EVENT" in TrialDf['name'].values:
        var = "anticipatory"
    elif "CHOICE_MISSED_EVENT" in TrialDf['name'].values:
        var = "missed"
    elif "PREMATURE_CHOICE_EVENT" in TrialDf['name'].values:
        var = "premature"
    elif "REWARD_AUTODELIVERED_EVENT" in TrialDf['name'].values:
        var = "jackpot"
    elif "CHOICE_INCORRECT_EVENT" in TrialDf['name'].values:
        var = "incorrect"
    elif "CHOICE_MISSED_EVENT" in TrialDf['name'].values:
        var = "missed"
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

def has_any_reach(TrialDf):
    var_name = 'any_reach'

    if has_reach_left(TrialDf)[0] or has_reach_right(TrialDf)[0]:
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

def get_delay_rt(TrialDf):
    " First reach RT during ONLY the DELAY period, agnostic of chosen side or arm"
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
    " TODO "

    " Classifies type of reach (contra,ipsi) for given trial"
    var_name = 'reach_type'
    sides = ['left','right']
    spouts = ['SL','SR']
    var = []

    if has_choice(TrialDf):
        choice_rt = get_choice_rt(TrialDf)
        t_choice = TrialDf['t'][0] + choice_rt

        for spout, side in zip(spouts,sides):
            if get_chosen_side(TrialDf).values == side:
                
                d_left_paw = dlc_utils.calc_dist_bp_bp(DLCDf, 'PL', spout)
                d_right_paw = dlc_utils.calc_dist_bp_bp(DLCDf, 'PR', spout)

                var = 'stuff'

    else:
        var = np.NaN

    return pd.Series(var, name=var_name)

def get_bias(TrialDf):
    var_name = "bias"
    try:
        Df = TrialDf.groupby('var').get_group(var_name)
        var = Df.iloc[0]['value']
    except KeyError:
        var = np.NaN

    return pd.Series(var, name=var_name)

def rew_collected(TrialDf):
    " Only works for autodelivered rewards"
    var_name = "rew_collect"

    if "REWARD_COLLECTED_EVENT" in TrialDf['name'].values:
        var = True
    else:
        var = False

    return pd.Series(var, name=var_name) 

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
        if has_choice(TrialDf).values:

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

def get_anticip_reach_outcome(TrialDf):
    var_name = 'anticip_reach_outcome'
    var = np.nan

    sl = event_slice(TrialDf, 'CHOICE_STATE', 'REWARD_EVENT')
    
    events = sl['name'].values
    choice = None

    if "REACH_LEFT_ON" in events and "REACH_RIGHT_ON" in events:
        ix_l = sl[sl['name'] == "REACH_LEFT_ON"].index[0]
        ix_r = sl[sl['name'] == "REACH_RIGHT_ON"].index[0]
        if ix_l < ix_r:
            choice = 'left'
        else:
            choice = 'right'

    if "REACH_LEFT_ON" in events:
        ix = sl[sl['name'] == "REACH_LEFT_ON"].index[0]
        choice = 'left'

    if "REACH_RIGHT_ON" in events:
        ix = sl[sl['name'] == "REACH_RIGHT_ON"].index[0]
        choice = 'right'

    if choice is not None:
        # correct_side = SessionDf.loc[i]['correct_side']
        correct_side = get_correct_side(TrialDf).values[0]
        if correct_side == choice:
            var = 'correct'
        else:
            var = 'incorrect'

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

def get_bias(TrialDf):
    var_name = "bias"
    try:
        Df = TrialDf.groupby('var').get_group(var_name)
        var = Df.iloc[0]['value'].astype('bool')
    except KeyError:
        var = np.NaN
        
    return pd.Series(var, name=var_name)

def get_autodeliver_trial(TrialDf):
    var_name = "autodeliver_rewards"
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

def get_reward_collection_rt(TrialDf):
    var_name = "reward_collection_rt"
    side = get_correct_side(TrialDf).values[0]
    try:
        Df = event_slice(TrialDf, "REWARD_%s_VALVE_ON" % side.upper(), "REWARD_COLLECTED_EVENT")
        var = Df.iloc[-1]['t'] - Df.iloc[0]['t']
    except IndexError:
        var = np.NaN
    return pd.Series(var, name=var_name)

    """
    ######## ######## ##     ## ########     ##     ##    ###    ########   ######
       ##    ##       ###   ### ##     ##    ###   ###   ## ##   ##     ## ##    ##
       ##    ##       #### #### ##     ##    #### ####  ##   ##  ##     ## ##
       ##    ######   ## ### ## ########     ## ### ## ##     ## ########   ######
       ##    ##       ##     ## ##           ##     ## ######### ##              ##
       ##    ##       ##     ## ##           ##     ## ##     ## ##        ##    ##
       ##    ######## ##     ## ##           ##     ## ##     ## ##         ######
    """


def get_trial_type(TrialDf):
    var_name = "this_trial_type"
    try:
        Df = TrialDf.groupby('var').get_group(var_name)
        var = Df.iloc[0]['value']
    except KeyError:
        var = np.NaN

    return pd.Series(var, name=var_name)

def get_delay(TrialDf):
    var_name = "this_delay"
    try:
        Df = TrialDf.groupby('var').get_group(var_name)
        var = Df.iloc[0]['value']
    except KeyError:
        var = np.NaN

    return pd.Series(var, name=var_name)

def get_reward_magnitude(TrialDf):
    var_name = "reward_magnitude"
    try:
        Df = TrialDf.groupby('var').get_group(var_name)
        var = Df.iloc[0]['value']
    except KeyError:
        var = np.NaN

    return pd.Series(var, name=var_name)

def get_reward_time(TrialDf):
    var_name = "reward_collection_time"
    # event = "REWARD_COLLECTED_EVENT"
    event = "REWARD_EVENT"
    if event in TrialDf['name'].values:
        try:
            Df = TrialDf.groupby('name').get_group(event)
            var = Df.iloc[0]['t']
        except KeyError:
            var = np.NaN
    else:
        var = np.NaN

    return pd.Series(var, name=var_name)