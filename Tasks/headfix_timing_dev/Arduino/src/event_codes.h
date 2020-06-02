// Codes template file
// all codes are const unsigned int

// NAMING CONVENTIONS

// _STATE for state machine states (entry)
// _ON _OFF are for events that have a start and a stop (spans)
// _EVENT are for actual events (time stamps)

// STATES
const unsigned int INI_STATE = 0;
const unsigned int TRIAL_AVAILABLE_STATE = 1;
const unsigned int PRESENT_INTERVAL_STATE = 2;
const unsigned int CHOICE_STATE = 3;
const unsigned int REWARD_AVAILABLE_STATE = 4;
const unsigned int TIMEOUT_STATE = 5;
const unsigned int ITI_STATE = 6;

// SPANS
const unsigned int LICK_ON = 10;
const unsigned int LICK_OFF = 11;
const unsigned int REWARD_VALVE_ON = 12;
const unsigned int REWARD_VALVE_OFF = 13;

// EVENTS

// trials and their possible outcomes
const unsigned int TRIAL_ENTRY_EVENT = 20;
const unsigned int TRIAL_ABORTED_EVENT = 21;
const unsigned int CHOICE_MISSED_EVENT = 22;
const unsigned int CHOICE_WRONG_EVENT = 23;
const unsigned int CHOICE_CORRECT_EVENT = 24;

// reward related
const unsigned int REWARD_AVAILABLE_EVENT = 30;
const unsigned int REWARD_COLLECTED_EVENT = 31;
const unsigned int REWARD_MISSED_EVENT = 33;

// choice related
const unsigned int CHOICE_LEFT_EVENT = 40;
const unsigned int CHOICE_RIGHT_EVENT = 41;

// stim and cue stuff
const unsigned int FIRST_TONE_EVENT = 50;
const unsigned int SECOND_TONE_EVENT = 51;