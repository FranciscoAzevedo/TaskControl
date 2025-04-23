// mapping: N/S/W/E corresponds to poke 0/1/2/3 in board

// ODOR VALVES - poke valve pins used for odor instead of water 
const int ODOR_NORTH_VALVE_PIN = 50; // HOW?
const int ODOR_SOUTH_VALVE_PIN = 4; // HOW?

// WATER VALVES AND PUMP
const int REWARD_WEST_VALVE_PIN = 45; // LZR1 - WEST to VALVE1
const int REWARD_EAST_VALVE_PIN = 46; // LZR2 - EAST to VALVE2
const int REWARD_PUMP_PIN = 53; // "STIM"

// POKES
const int POKE_NORTH_PIN = 13; // poke0 receiver
const int POKE_SOUTH_PIN = 11; // poke1
const int POKE_WEST_PIN = 8; // poke2
const int POKE_EAST_PIN = 5; // poke3

// LIGHT
const int POKE_NORTH_LED = 48; // poke0 led
const int POKE_SOUTH_LED = 12; // poke1
const int POKE_WEST_LED = 9; // poke2
const int POKE_EAST_LED = 6; // poke3

const int BACKGROUND_LIGHT = 4; // 

// SOUND 
const int SPEAKERS_PIN = 3; // both speakers connected to same pin

// SYNC
const int CAM_SYNC_PIN = 30; // HOW TO DO THIS?