// Parameters from Thiago Gouvea's eLife paper
unsigned long tone_dur = 150;
unsigned long tone_freq = 7500; 

unsigned long ITI_dur_min = 8500;
unsigned long ITI_dur_max = 11500;
unsigned long timeout_dur = 10000;
unsigned long choice_dur = 5000;

// Parameters to be controlled across task progression 
int autodeliver_rewards = 0;
int left_short = 1;
unsigned long reward_magnitude = 10;
int no_intervals = 1; // 1 means easiest, 3 means whole set

int trial_autostart = 0;
