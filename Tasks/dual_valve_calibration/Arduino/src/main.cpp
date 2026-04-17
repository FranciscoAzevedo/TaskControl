#include <Arduino.h>
#include <Tone.h>
#include <Adafruit_NeoPixel.h>

#include <event_codes.h> // <>?
#include "interface.cpp"
#include "pin_map.h"
#include "logging.cpp"

// Parameters for pump and valve calibration
const int pumpToggleDur = 3; // ms per toggle
int valvePin = -1; // placeholder, will be set in calibration function
const unsigned long rewardValveDur = 2000; // ms valve open duration
unsigned long pump_start_time = 0;

// FSM variables
int last_state = -1; // whatever other state
unsigned long max_future = 4294967295UL;
unsigned long t_state_entry = max_future;

void state_entry_common(){
    last_state = current_state;
    t_state_entry = now();
    log_code(current_state);
}

void togglePump(int toggles) {
    for (int i = 0; i < toggles; i++) {
        digitalWrite(REWARD_PUMP_PIN, HIGH);
        delay(pumpToggleDur);
        digitalWrite(REWARD_PUMP_PIN, LOW);
        delay(pumpToggleDur);
    }
}

void calibrateValve(int side) {
    log_msg("Starting calibration for ");
    log_int("side",side);

    if (side == 1) {
        valvePin = REWARD_WEST_VALVE_PIN;
    } else if (side == 2) {
        valvePin = REWARD_EAST_VALVE_PIN;
    }
    
    for (int i = 0; i < numPumpTriggers; i++) {
        pump_start_time = now();

        digitalWrite(valvePin, HIGH);
        log_msg("Valve open");
        while (now() - pump_start_time < rewardValveDur) {
            togglePump(targetToggles);
        }
        
        digitalWrite(valvePin, LOW);
        log_msg("Valve closed");

        // Wait a bit before the next trigger
        delay(waitBetweenPumps);
    }
    log_msg("Calibration complete for ");
    log_int("side",side);

}

void finite_state_machine(){
    switch (current_state){
        
        case STANDBY_STATE:
            if (calibrate_west == true) {
                current_state = CALIB_WEST_STATE;
            } else if (calibrate_east == true) {
                current_state = CALIB_EAST_STATE;
            }
            break;

        case CALIB_WEST_STATE:
            if (current_state != last_state){
                state_entry_common();
                calibrateValve(1);
                calibrate_west = false; // reset flag
                current_state = DONE_STATE;
            }
            break;

        case CALIB_EAST_STATE:
            if (current_state != last_state){
                state_entry_common();
                calibrateValve(2);
                calibrate_east = false; // reset flag
                current_state = DONE_STATE;
            }
            break;

        case DONE_STATE:
            if (current_state != last_state){
                state_entry_common();
                log_msg("Valve calibration finished.");
                calibrate_west = false;
                calibrate_east = false;
                current_state = STANDBY_STATE;
            }
            break;
    }
}

void setup() {
    Serial.begin(115200);
    pinMode(REWARD_PUMP_PIN, OUTPUT);
    pinMode(REWARD_WEST_VALVE_PIN, OUTPUT);
    pinMode(REWARD_EAST_VALVE_PIN, OUTPUT);
    delay(1000);
}

void loop() {
    if (run == true){
        // execute state machine(s)
        finite_state_machine();
    }

    // serial communication with main PC
    getSerialData();
    processSerialData();
}
