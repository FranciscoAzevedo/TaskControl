// Codes template file

// NAMING CONVENTIONS

// _STATE for state machine states (entry)
// _ON _OFF are for events that have a start and a stop (spans)
// _EVENT are for actual events (time stamps)

// N/S/E/W refer to world-coordinates, used on immutable things
// such as port locations and reward location

// L/R refer to body-coordinates, used on things depending on the animal

// STATES
const unsigned int INI_STATE = 0;
const unsigned int TRIAL_AVAILABLE_STATE = 1;
const unsigned int CHOICE_STATE = 2;
const unsigned int REWARD_STATE = 3;
const unsigned int ITI_STATE = 4;

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

unsigned int ODOR_NORTH_VALVE_ON = 22;
unsigned int ODOR_NORTH_VALVE_OFF = 23;
unsigned int ODOR_SOUTH_VALVE_ON = 24;
unsigned int ODOR_SOUTH_VALVE_OFF = 25;

// EVENTS

// trials and their possible outcomes
unsigned int TRIAL_AVAILABLE_EVENT = 26;
unsigned int TRIAL_SUCCESSFUL_EVENT = 31;
unsigned int TRIAL_UNSUCCESSFUL_EVENT = 32;

unsigned int CHOICE_MISSED_EVENT = 33;
unsigned int CHOICE_INCORRECT_EVENT = 34;
unsigned int CHOICE_CORRECT_EVENT = 35;

// reward related
unsigned int REWARD_WEST_EVENT = 36;
unsigned int REWARD_EAST_EVENT = 37;

// choice related
unsigned int CHOICE_EVENT = 40;
unsigned int CHOICE_WEST_EVENT = 42;
unsigned int CHOICE_EAST_EVENT = 43;

// stim and cue stuff
unsigned int LIGHT_WEST_CUE_EVENT = 54;
unsigned int LIGHT_EAST_CUE_EVENT = 55;