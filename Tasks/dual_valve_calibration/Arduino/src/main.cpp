#include <Arduino.h>
#include <Tone.h>
#include "pin_map.h"

// Parameters for pump and valve calibration
const int pumpToggleDur = 3; // ms per toggle
const unsigned long rewardValveDur = 2000; // ms valve open duration

// Pin definitions (update as needed)
const int PUMP_PIN = 10; // Example pin for pump
const int REWARD_VALVE_WEST_PIN = 8;
const int REWARD_VALVE_EAST_PIN = 9;

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
        togglePump(targetToggles);
        openValve(valvePin, rewardValveDur);
        Serial.println("Valve open");
        delay(waitBetweenPumps);
    }
    Serial.print("Calibration complete for ");
    Serial.println(side);
}

void setup() {
    Serial.begin(115200);
    pinMode(PUMP_PIN, OUTPUT);
    pinMode(REWARD_VALVE_WEST_PIN, OUTPUT);
    pinMode(REWARD_VALVE_EAST_PIN, OUTPUT);
    delay(10);

    calibrateValve(REWARD_VALVE_WEST_PIN, "West");
    calibrateValve(REWARD_VALVE_EAST_PIN, "East");
    Serial.println("Valve calibration task finished.");
}

void loop() {
    // Task is finished, nothing to do
}
