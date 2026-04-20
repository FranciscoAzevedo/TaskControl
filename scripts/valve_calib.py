# %%
%matplotlib qt5
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

"""
    Method to calibrate valve
    1 - Run valve_calibration task with increments of reward volume (30,60,..)
    2 - Weight the resulting water that comes out (register to CSV) and refill to 5ml
    3 - Run this script to get slope (m) which is valve_ul_ms value
"""

Df = pd.read_csv('valve_calib_ego_allo_newbox.csv')

Df['weight'] = Df['weight']*1000
Df['w_per_rep'] = Df['weight'].values / Df['reps'].values

no_triggers = Df['time'].values

%matplotlib qt5

def lin(x,m,b):
    return m*x+b
fig, axes = plt.subplots()

fvec = np.linspace(20,80,5)
for i, side in enumerate(['W','E']):
    df = Df.groupby('side').get_group(side)
    m, b = stats.linregress(df['time'].values, df['w_per_rep'].values)[:2]
    dots, = axes.plot(df['time'].values, df['w_per_rep'].values,'o',label=side)
    axes.plot(fvec, lin(fvec, m, b),lw=1,linestyle=':',color=dots.get_color())
    print("valve_ul_ms for side %s : %.2e" % (side, m))
axes.legend()
# %%
