// Codes template file
// all codes are const unsigned int

// NAMING CONVENTIONS

// _STATE for state machine states (entry)
// _ON _OFF are for events that have a start and a stop (spans)
// _EVENT are for actual events (time stamps)

// N/S/E/W refer to world-coordinates, used on immutable things
// such as port locations and reward location

// L/R refer to body-coordinates, used on movements dependent on the animal

// STATES
const unsigned int INI_STATE = 0;
const unsigned int TRIAL_AVAILABLE_STATE = 1;
const unsigned int TRIAL_ENTRY_STATE = 2;
const unsigned int PRESENT_INTERVAL_STATE = 3;
const unsigned int CHOICE_STATE = 4;
const unsigned int REWARD_STATE = 5;
const unsigned int ITI_STATE = 6;
const unsigned int TIMEOUT_STATE = 7;

// SPANS
unsigned int POKE_NORTH_IN = 8;
unsigned int POKE_NORTH_OUT = 9;
unsigned int POKE_SOUTH_IN = 10;
unsigned int POKE_SOUTH_OUT = 11;
unsigned int POKE_WEST_IN = 12;
unsigned int POKE_WEST_OUT = 13;
unsigned int POKE_EAST_IN = 14;
unsigned int POKE_EAST_OUT = 15;

unsigned int WATER_PUMP_ON = 16;
unsigned int WATER_PUMP_OFF = 17;

unsigned int WATER_WEST_VALVE_ON = 18;
unsigned int WATER_WEST_VALVE_OFF = 19;
unsigned int WATER_EAST_VALVE_ON = 20;
unsigned int WATER_EAST_VALVE_OFF = 21;

// jitter
unsigned int JITTER_IN = 23;
unsigned int JITTER_OUT = 24;


// EVENTS

// trials and outcomes
unsigned int TRIAL_AVAILABLE_EVENT = 26;
unsigned int TRIAL_ENTRY_EVENT = 27;
unsigned int TRIAL_ENTRY_NORTH_EVENT = 28;
unsigned int TRIAL_ENTRY_SOUTH_EVENT = 29;
unsigned int BROKEN_FIXATION_EVENT = 30;
unsigned int TRIAL_SUCCESSFUL_EVENT = 31;
unsigned int TRIAL_UNSUCCESSFUL_EVENT = 32;

unsigned int CHOICE_MISSED_EVENT = 33;
unsigned int CHOICE_INCORRECT_EVENT = 34;
unsigned int CHOICE_CORRECT_EVENT = 35;
unsigned int TRIAL_AVAILABLE_TIMEOUT_EVENT = 38;

// reward 
unsigned int REWARD_WEST_EVENT = 36;
unsigned int REWARD_EAST_EVENT = 37;

// choice 
unsigned int CHOICE_EVENT = 40;
unsigned int CHOICE_WEST_EVENT = 42;
unsigned int CHOICE_EAST_EVENT = 43;
unsigned int CHOICE_LONG_EVENT = 44; 
unsigned int CHOICE_SHORT_EVENT = 45;
unsigned int CHOICE_LEFT_EVENT = 46;
unsigned int CHOICE_RIGHT_EVENT = 47;

// stim
unsigned int FIRST_TIMING_CUE_EVENT = 50;
unsigned int SECOND_TIMING_CUE_EVENT = 51;
unsigned int INIT_POKEOUT_EVENT = 52;

// cue
unsigned int LIGHT_NORTH_CUE_EVENT = 55;
unsigned int LIGHT_SOUTH_CUE_EVENT = 56;
unsigned int LIGHT_WEST_CUE_EVENT = 57;
unsigned int LIGHT_EAST_CUE_EVENT = 58;