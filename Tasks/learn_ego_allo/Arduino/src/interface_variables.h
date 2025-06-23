// Parameters to be controlled across task progression 
unsigned long ITOI_dur = 9000; // Inter trial ONSET interval - from cooling paper
unsigned long timeout_dur = 10000; // from cooling paper
unsigned long choice_dur = 5000;

unsigned long mean_fix_dur = 1500; // higher since they learn to fixate before
unsigned long inc_fix_dur = 20;
unsigned long dec_fix_dur = 10;
unsigned long sigma_fix = 200;

unsigned long init_port_blocks = 0; // 0-false, 1-true
unsigned long port_dur_min = 15;
unsigned long port_dur_max = 30;

unsigned long block_dur_min = 40;
unsigned long block_dur_max = 80;

// full task
int learning = 1; //  0-false (steady state), 1-true (learning) 
int no_intervals = 1; // 1 means easiest, 3 means whole set - only applies to steady state
