// Parameters to be controlled across task progression 
unsigned long ITOI_dur = 9000; // Inter trial ONSET interval - 9s in cooling paper
unsigned long timeout_dur = 10000; // from cooling paper
unsigned long choice_dur = 8000;

unsigned long init_port_blocks = 0; // 0-false, 1-true
unsigned long port_dur_min = 15;
unsigned long port_dur_max = 30;

unsigned long block_dur_min = 40;
unsigned long block_dur_max = 60;

int blind_eye_ON = 0; // 1 means true - ON

// full task
int use_correction_loops = 0; // 0 means no correction loops, 1 means correction loops are used
int corr_loop_entry = 3; // threshold for entering correction loop

float p_cued = 1; // probability of cued trial - 1 means 100% cued, 0 means 0% cued
int no_intervals = 3; // 1 means easiest, 3 means whole set - only applies to steady state
