unsigned long tone_dur = 50;
unsigned long buzz_dur = 50;
unsigned long trial_entry_buzz_dur = 5;
unsigned long buzz_center_freq = 235;
unsigned long buzz_freq_sep = 35;
int led_hsv = 180;
int led_brightness = 100;

unsigned long ITI_dur_min = 8500;
unsigned long ITI_dur_max = 11500;
unsigned long timeout_dur = 6000;
unsigned long choice_dur = 3000;

int cue_on_rewarded_reach = 0;
int cue_on_reach = 0;
int present_init_cue = 0;
int allow_mistakes = 1;
int autodeliver_rewards = 1;
int left_short = 1;
unsigned long reward_magnitude = 4;

float valve_ul_ms_left = 0.01;
float valve_ul_ms_right = 0.01;

int n_warmup_trials = 10;
int n_max_miss_trials = 10;
int correction_loops = 1;
int corr_loop_entry = 3;
int corr_loop_exit = 2;
unsigned long reach_block_dur = 1000;
unsigned long min_grasp_dur = 20;
int trial_autostart = 1;
float p_timing_trial = 0.0;

float bias = 0.5;
// float contrast = 1.0;

unsigned long gap = 500;