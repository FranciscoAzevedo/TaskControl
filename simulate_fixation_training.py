#  %%
import numpy as np
import matplotlib.pyplot as plt

# Parameters from interface_variables.h
ITOI_dur = 9000
timeout_dur = 5000
choice_dur = 20000
t_init_max = 60000

mean_fix_dur_init = 0.3  # seconds, start with 0.3s for simulation
sigma_fix_dur_frac = 0.25

# Simulation parameters
n_sessions = 10
trials_per_session = 300

# For plotting
all_mean_fix = []
all_sigma = []


for session in range(n_sessions):
    mean_fix_dur = mean_fix_dur_init
    mean_fix_list = []
    sigma_list = []
    for trial in range(trials_per_session):
        sigma = mean_fix_dur * sigma_fix_dur_frac

        # Probability of breaking fixation: 30% at 0.3s, 50% at 2s
        p_break = 0.3 + ((mean_fix_dur - 0.3) / (2.0 - 0.3)) * (0.7 - 0.5)
        p_break = np.clip(p_break, 0.5, 0.7)

        # inc_fix_dur: 10ms at 0.3s, 50ms at 2s (linear)
        inc_fix_dur = 0.01 + ((mean_fix_dur - 0.3) / (2.0 - 0.3)) * (0.05 - 0.002)
        # dec_fix_dur: 2ms at 0.3s, 10ms at 2s (linear)
        dec_fix_dur = 0.002 + ((mean_fix_dur - 0.3) / (2.0 - 0.3)) * (0.01 - 0.001)

        broke_fix = np.random.rand() < p_break
        if broke_fix:
            mean_fix_dur = max(mean_fix_dur - dec_fix_dur, 0.05)
        else:
            mean_fix_dur = mean_fix_dur + inc_fix_dur
        mean_fix_list.append(mean_fix_dur)
        sigma_list.append(sigma)
    all_mean_fix.append(mean_fix_list)
    all_sigma.append(sigma_list)

# Convert to numpy arrays for easier plotting
all_mean_fix = np.array(all_mean_fix)
all_sigma = np.array(all_sigma)

# Plot
plt.figure(figsize=(10, 6))
trials = np.arange(trials_per_session)
for i in range(n_sessions):
    plt.plot(trials, all_mean_fix[i], color='C0', alpha=0.7, linewidth=2)
    plt.fill_between(trials, all_mean_fix[i] - all_sigma[i], all_mean_fix[i] + all_sigma[i],
                        color='C0', alpha=0.15)
plt.title('Simulated Training Sessions: mean_fix_dur and sigma')
plt.xlabel('Trial')
plt.ylabel('Fixation Duration (s)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
