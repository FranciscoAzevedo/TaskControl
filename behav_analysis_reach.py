#%matplotlib qt5
#%load_ext autoreload
#%autoreload 2

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 166
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm 
from matplotlib import patches
from scipy import signal

from sklearn.linear_model import LogisticRegression

import pandas as pd
import itertools
from pathlib import Path
import scipy as sp
import numpy as np
import seaborn as sns
from tqdm import tqdm
import utils
import os
import calendar

import behavior_analysis_utils as bhv
import behavior_plotters as bhv_plt

"""
 
 ########  ########    ###    ########  
 ##     ## ##         ## ##   ##     ## 
 ##     ## ##        ##   ##  ##     ## 
 ########  ######   ##     ## ##     ## 
 ##   ##   ##       ######### ##     ## 
 ##    ##  ##       ##     ## ##     ## 
 ##     ## ######## ##     ## ########  
 
"""

# Multi session loading
animal_folder = utils.get_folder_dialog()
task_name = ['learn_to_reach']
SessionsDf = utils.get_sessions(animal_folder)

log_paths = [Path(path)/'arduino_log.txt' for path in SessionsDf['path']]

for log_path in tqdm(log_paths):

    path = log_path.parent 
    print('\n')
    print(path)

    LogDf = bhv.get_LogDf_from_path(log_path)

    # %% make SessionDf - slice into trials
    TrialSpans = bhv.get_spans_from_names(LogDf, "TRIAL_ENTRY_STATE", "ITI_STATE")

    TrialDfs = []
    for i, row in tqdm(TrialSpans.iterrows(),position=0, leave=True):
        TrialDfs.append(bhv.time_slice(LogDf, row['t_on'], row['t_off']))

    metrics = (bhv.get_start, bhv.get_stop, bhv.get_correct_zone)
    SessionDf = bhv.parse_trials(TrialDfs, metrics)

    