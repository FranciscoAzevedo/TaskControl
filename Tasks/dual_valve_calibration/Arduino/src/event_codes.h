// Codes template file
// all codes are const unsigned int

// NAMING CONVENTIONS

// _STATE for state machine states (entry)
// _ON _OFF are for events that have a start and a stop (spans)
// _EVENT are for actual events (time stamps)

// STATES
const unsigned int STANDBY_STATE = 3;
const unsigned int CALIB_WEST_STATE = 1;
const unsigned int CALIB_EAST_STATE = 2;
const unsigned int DONE_STATE = 0;

// SPANS
const unsigned int REWARD_VALVE_ON = 12;
const unsigned int REWARD_VALVE_OFF = 13;



