#include <Arduino.h>
#include <Tone.h>
#include "pin_map.h"
#include "interface_variables.h"
#include "event_codes.h"

// Parameters for pump and valve calibration
const int pumpToggleDur = 3; // ms per toggle
const unsigned long rewardValveDur = 2000; // ms valve open duration

// Pin definitions (update as needed)
const int PUMP_PIN = 10; // Example pin for pump
const int REWARD_VALVE_WEST_PIN = 8;
const int REWARD_VALVE_EAST_PIN = 9;

// FSM variables
unsigned long max_future = 4294967295UL;
int current_state = STANDBY_STATE;
int last_state = -1;
unsigned long t_state_entry = max_future;
bool calibrate_west_done = false;
bool calibrate_east_done = false;

unsigned long now() { return millis(); }

void log_code(int code) {
    Serial.println(code);
}

void state_entry_common(){
    last_state = current_state;
    t_state_entry = now();
    log_code(current_state);
}

void togglePump(int toggles) {
    for (int i = 0; i < toggles; i++) {
        digitalWrite(PUMP_PIN, HIGH);
        delay(pumpToggleDur);
        digitalWrite(PUMP_PIN, LOW);
        delay(pumpToggleDur);
    }
}

void openValve(int valvePin, unsigned long duration) {
    digitalWrite(valvePin, HIGH);
    delay(duration);
    digitalWrite(valvePin, LOW);
}

void calibrateValve(int valvePin, const char* side) {
    Serial.print("Starting calibration for ");
    Serial.println(side);
    
    for (int i = 0; i < numPumpTriggers; i++) {
        openValve(valvePin, rewardValveDur);
        togglePump(targetToggles);
        Serial.println("Valve open");
        delay(waitBetweenPumps);
    }
    Serial.print("Calibration complete for ");
    Serial.println(side);
}

void finite_state_machine(){
    switch (current_state){
        
        case STANDBY_STATE:
            if (!calibrate_west_done && calibrate_west) {
                current_state = CALIB_WEST_STATE;
            } else if (!calibrate_east_done && calibrate_east) {
                current_state = CALIB_EAST_STATE;
            } else {
                current_state = DONE_STATE;
            }
            break;

        case CALIB_WEST_STATE:
            if (current_state != last_state){
                state_entry_common();
                calibrateValve(REWARD_VALVE_WEST_PIN, "West");
                calibrate_west_done = true;
                current_state = STANDBY_STATE;
            }
            break;

        case CALIB_EAST_STATE:
            if (current_state != last_state){
                state_entry_common();
                calibrateValve(REWARD_VALVE_EAST_PIN, "East");
                calibrate_east_done = true;
                current_state = STANDBY_STATE;
            }
            break;

        case DONE_STATE:
            if (current_state != last_state){
                state_entry_common();
                Serial.println("Valve calibration task finished.");
            }
            break;
    }
}

void setup() {
    Serial.begin(115200);
    pinMode(PUMP_PIN, OUTPUT);
    pinMode(REWARD_VALVE_WEST_PIN, OUTPUT);
    pinMode(REWARD_VALVE_EAST_PIN, OUTPUT);
    delay(10);
}

void loop() {
    finite_state_machine();
}
