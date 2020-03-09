// Codes template file
// all codes are const unsigned int

// NAMING CONVENTIONS

// _STATE for state machine states (entry)
// _ON _OFF are for events that have a start and a stop (spans)
// _EVENT are for actual events (time stamps)

// STATES
const unsigned int INI_STATE = 0;
const unsigned int TRIAL_AVAILABLE_STATE = 1;
const unsigned int NO_REWARD_AVAILABLE_STATE = 2;
const unsigned int REWARD_AVAILABLE_STATE = 3;
const unsigned int ITI_STATE = 4;

// SPANS
const unsigned int LICK_ON = 10;
const unsigned int LICK_OFF = 11;
const unsigned int REWARD_VALVE_ON = 12;
const unsigned int REWARD_VALVE_OFF = 13;

// EVENTS
const unsigned int TRIAL_ENTRY_EVENT = 20;

const unsigned int REWARD_AVAILABLE_EVENT = 30;
const unsigned int REWARD_COLLECTED_EVENT = 31;
const unsigned int REWARD_MISSED_EVENT = 32;
const unsigned int NO_REWARD_AVAILABLE_EVENT = 33;


