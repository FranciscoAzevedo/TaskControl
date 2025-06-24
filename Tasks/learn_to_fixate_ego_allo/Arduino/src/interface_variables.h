// Parameters to be controlled across task progression 
unsigned long ITOI_dur = 9000; // // Inter trial ONSET interval - from cooling paper
unsigned long timeout_dur = 5000; // from cooling paper
unsigned long choice_dur = 20000;
unsigned long t_init_max = 60000; // max time to wait for trial entry

unsigned long mean_fix_dur = 20; // starting point
unsigned long inc_fix_dur = 10; // go from 10,20,50
unsigned long dec_fix_dur = 5; // go from 5,10,20
unsigned long sigma_fix = 5;

unsigned long init_port_blocks = 0; // 0-false, 1-true
unsigned long port_dur_min = 15;
unsigned long port_dur_max = 30;
