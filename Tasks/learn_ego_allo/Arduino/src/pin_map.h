// mapping: N/S/W/E corresponds to poke 0/1/2/3 in baseboard
#define NUM_POKES 4

// ODOR VALVES - the neutral flow carrier is the normally open path on the 3-way valves
const int ODOR1_NORTH_VALVE_PIN = 38; // these 4 connections come soldered from the PCB, not via terminal block
const int ODOR2_NORTH_VALVE_PIN = 39; // 
const int ODOR1_SOUTH_VALVE_PIN = 40; // 
const int ODOR2_SOUTH_VALVE_PIN = 41; // 

// WATER VALVES AND PUMP
const int REWARD_WEST_VALVE_PIN = 10; // poke2 valve
const int REWARD_EAST_VALVE_PIN = 7; // poke3 valve
const int REWARD_PUMP_PIN = 53; // "STIM" on diagram

// POKES  (redundant definitions for clarity)
const int POKE_NORTH_PIN = 13; // poke0 receiver
const int POKE_SOUTH_PIN = 11; // poke1
const int POKE_WEST_PIN = 8; // poke2
const int POKE_EAST_PIN = 5; // poke3

const unsigned int POKES_PINS[NUM_POKES] = {
    13, // poke0
    11, // poke1
    8,  // poke2
    5   // poke3  
};

// SOUND AND LIGHT
const int BCKGND_LIGHTS_PIN = 44; // "LZR" on diagram

const unsigned int POKES_LED_PINS[NUM_POKES] = {
    48, // poke0
    12, // poke1
    9,  // poke2
    6   // poke3  
};

const int SPEAKER_WEST_PIN = 3; // speaker west
const int SPEAKER_EAST_PIN = 49; // speaker east

// SYNC
const int CAM_SYNC_PIN = 45; // "LZR1" on diagram
