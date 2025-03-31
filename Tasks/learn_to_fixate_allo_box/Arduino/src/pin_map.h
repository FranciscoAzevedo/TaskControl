// mapping: N/S/W/E corresponds to poke 0/1/2/3 in board

// ODOR VALVES - poke valve pins used for odor instead of water 
const int ODOR_NORTH_VALVE_PIN = 50; // poke0 valve
const int ODOR_SOUTH_VALVE_PIN = 4; // poke1 valve

// WATER VALVES AND PUMP
const int REWARD_WEST_VALVE_PIN = 10; // poke2 valve
const int REWARD_EAST_VALVE_PIN = 7; // poke3 valve
const int REWARD_PUMP_PIN = 53; // "STIM"

// POKES
const int POKE_NORTH_PIN = 13; // poke0 receiver
const int POKE_SOUTH_PIN = 11; // poke1
const int POKE_WEST_PIN = 8; // poke2
const int POKE_EAST_PIN = 5; // poke3

// SOUND AND LIGHT
const int POKE_NORTH_LED = 48; // poke0 led
const int POKE_SOUTH_LED = 12; // poke1
const int POKE_WEST_LED = 9; // poke2
const int POKE_EAST_LED = 6; // poke3

const int SPEAKERS_PIN = 3; // both speakers connected to same pin

// SYNC
const int CAM_SYNC_PIN = 30; // HOW TO DO THIS?
const int TRIAL_INIT_PIN = 31; // HOW TO DO THIS?