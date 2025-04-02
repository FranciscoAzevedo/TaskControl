// Parameters from Thiago Gouvea's eLife paper
unsigned long tone_dur = 150;
unsigned long tone_freq = 7500;
unsigned long reward_tone_freq = 1750;
unsigned long reward_magnitude = 10;
unsigned long reward_valve_dur = 3000; // more than enough for pump

// Parameters to be controlled across task progression 
unsigned long ITI_dur_min = 8500;
unsigned long ITI_dur_max = 11500;
unsigned long timeout_dur = 10000;
unsigned long choice_dur = 5000;

unsigned long min_fix_dur = 0;
unsigned long inc_fix_dur = 10;
unsigned long dec_fix_dur = 5;

unsigned long block_dur_min = 40;
unsigned long block_dur_max = 80;

int left_long = 1; // this is an egocentric mapping

int no_intervals = 1; // 1 means easiest, 3 means whole set
