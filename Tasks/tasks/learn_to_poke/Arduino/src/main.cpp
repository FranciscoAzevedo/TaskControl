#include <Arduino.h>
#include <Tone.h>
#include <FastLED.h>

#include <event_codes.h> // <>?
#include "interface.cpp"
#include "pin_map.h"
#include "logging.cpp"

/*
########  ########  ######  ##          ###    ########     ###    ######## ####  #######  ##    ##  ######
##     ## ##       ##    ## ##         ## ##   ##     ##   ## ##      ##     ##  ##     ## ###   ## ##    ##
##     ## ##       ##       ##        ##   ##  ##     ##  ##   ##     ##     ##  ##     ## ####  ## ##
##     ## ######   ##       ##       ##     ## ########  ##     ##    ##     ##  ##     ## ## ## ##  ######
##     ## ##       ##       ##       ######### ##   ##   #########    ##     ##  ##     ## ##  ####       ##
##     ## ##       ##    ## ##       ##     ## ##    ##  ##     ##    ##     ##  ##     ## ##   ### ##    ##
########  ########  ######  ######## ##     ## ##     ## ##     ##    ##    ####  #######  ##    ##  ######
*/

int last_state = -1; // whatever other state
unsigned long max_future = 4294967295; // 2**32 -1
unsigned long t_state_entry = max_future;
unsigned long grace_period = 50; // ms

// because its easier to compare numbers than strings
int north = 8;
int south = 2;
int west = 4;
int east = 6;

// ego centric coords
int left = 7;
int right = 9;

// non-init variables
float r; // for random processes
unsigned long t_poke_remain;
unsigned long this_ITI_dur;
int choice;
int correct_movement;
int correct_side;
int init_port;

// // bias related
// int last_correct_movement = left;

// float bias = 0.5; // exposed in interface_variables.h
// int n_choices_left = 1;
// int n_choices_right = 1;

// void update_bias(){
//     // 0 = left bias, 1 = right bias
//     bias = (float) n_choices_right / (n_choices_left + n_choices_right);
// }

/*
 ######  ######## ##    ##  ######   #######  ########   ######
##    ## ##       ###   ## ##    ## ##     ## ##     ## ##    ##
##       ##       ####  ## ##       ##     ## ##     ## ##
 ######  ######   ## ## ##  ######  ##     ## ########   ######
      ## ##       ##  ####       ## ##     ## ##   ##         ##
##    ## ##       ##   ### ##    ## ##     ## ##    ##  ##    ##
 ######  ######## ##    ##  ######   #######  ##     ##  ######
*/
bool is_poking_north = false;
bool poke_north = false;

bool is_poking_south = false;
bool poke_south = false;

bool is_poking_west = false;
bool poke_west = false;

bool is_poking_east = false;
bool poke_east = false;

bool is_poking = false;
unsigned long t_last_poke_on = max_future;
unsigned long t_last_poke_off = max_future;

void read_pokes(){
    // north
    poke_north = digitalRead(POKE_NORTH_PIN);
    if (is_poking_north == false && poke_north == true){
        log_code(POKE_NORTH_IN);
        is_poking_north = true;
        t_last_poke_on = now();
    }

    if (is_poking_north == true && poke_north == false){
        log_code(POKE_NORTH_OUT);
        is_poking_north = false;
        t_last_poke_off = now();
    }

    // south
    poke_south = digitalRead(POKE_SOUTH_PIN);
    if (is_poking_south == false && poke_south == true){
        log_code(POKE_SOUTH_IN);
        is_poking_south = true;
        t_last_poke_on = now();
    }

    if (is_poking_south == true && poke_south == false){
        log_code(POKE_SOUTH_OUT);
        is_poking_south = false;
        t_last_poke_off = now();
    }

    // west
    poke_west = digitalRead(POKE_WEST_PIN);
    if (is_poking_west == false && poke_west == true){
        log_code(POKE_WEST_IN);
        is_poking_west = true;
        t_last_poke_on = now();
    }

    if (is_poking_west == true && poke_west == false){
        log_code(POKE_WEST_OUT);
        is_poking_west = false;
        t_last_poke_off = now();
    }

    // east
    poke_east = digitalRead(POKE_EAST_PIN);
    if (is_poking_east == false && poke_east == true){
        log_code(POKE_EAST_IN);
        is_poking_east = true;
        t_last_poke_on = now();
    }

    if (is_poking_east == true && poke_east == false){
        log_code(POKE_EAST_OUT);
        is_poking_east = false;
        t_last_poke_off = now();
    }

    is_poking = (is_poking_north || is_poking_south || is_poking_west || is_poking_east);
}

/*
##       ######## ########
##       ##       ##     ##
##       ##       ##     ##
##       ######   ##     ##
##       ##       ##     ##
##       ##       ##     ##
######## ######## ########
*/

void go_cue_west(){
    digitalWrite(POKE_WEST_LED, HIGH);
    log_code(LIGHT_WEST_CUE_EVENT);
}

void go_cue_east(){
    digitalWrite(POKE_EAST_LED, HIGH);
    log_code(LIGHT_EAST_CUE_EVENT);
}

// // LED strip related
// #define NUM_LEDS 2 // num of LEDs in strip
// CRGB leds[NUM_LEDS]; // Define the array of leds

// // LED blink controller related
// bool led_is_on[NUM_LEDS];
// bool switch_led_on[NUM_LEDS];
// unsigned long led_on_time[NUM_LEDS];
// unsigned long led_on_dur = 50;

// void led_blink_controller(){
//     // the controller: iterate over all LEDs and set their state accordingly
//     for (int i = 0; i < NUM_LEDS; i++){
//         if (led_is_on[i] == false && switch_led_on[i] == true){
//             // leds[i] = CRGB::Blue; // can be replaced with HSV maybe?
//             leds[i] = CHSV(led_hsv,255,led_brightness);
//             led_is_on[i] = true;
//             led_on_time[i] = now();
//             switch_led_on[i] = false;
//             FastLED.show();
//         }
//         // turn off if on for long enough
//         if (led_is_on[i] == true && now() - led_on_time[i] > led_on_dur){
//             leds[i] = CRGB::Black;
//             led_is_on[i] = false;
//             FastLED.show();
//         }
//     }
// }

// // access functions
// void lights_on(){
//     for (int i = 0; i < NUM_LEDS; i++){
//         leds[i] = CHSV(led_hsv,255,led_brightness);
//     }
//     FastLED.show();
// }

// void lights_off(){
//     for (int i = 0; i < NUM_LEDS; i++){
//         leds[i] = CRGB::Black;
//     }
//     FastLED.show();
// }

/*
 ######  ##     ## ########  ######
##    ## ##     ## ##       ##    ##
##       ##     ## ##       ##
##       ##     ## ######    ######
##       ##     ## ##             ##
##    ## ##     ## ##       ##    ##
 ######   #######  ########  ######
*/

// speaker
Tone tone_controller;

void trial_available_cue(){
    // turn both ports' lights ON
    digitalWrite(POKE_NORTH_LED, HIGH);
    log_code(LIGHT_NORTH_CUE_EVENT);
    digitalWrite(POKE_SOUTH_LED, HIGH);
    log_code(LIGHT_SOUTH_CUE_EVENT);
}

void sound_cue(){     
    tone_controller.play(tone_freq, tone_dur);
}

void reward_cue(){
    tone_controller.play(reward_tone_freq, tone_dur);
}

/*
adapted from  https://arduino.stackexchange.com/questions/6715/audio-frequency-white-noise-generation-using-arduino-mini-pro
*/
#define LFSR_INIT  0xfeedfaceUL
#define LFSR_MASK  ((unsigned long)( 1UL<<31 | 1UL <<15 | 1UL <<2 | 1UL <<1  ))

unsigned int generateNoise(){ 
    // See https://en.wikipedia.org/wiki/Linear_feedback_shift_register#Galois_LFSRs
    static unsigned long int lfsr = LFSR_INIT;
    if(lfsr & 1) { lfsr =  (lfsr >>1) ^ LFSR_MASK ; return(1);}
    else         { lfsr >>= 1;                      return(0);}
}

unsigned long error_cue_start = max_future;
unsigned long error_cue_dur = tone_dur * 1000; // to save instructions - work in micros
unsigned long lastClick = max_future;

void incorrect_choice_cue(){
    // white noise - blocking arduino for tone_dur
    error_cue_start = micros();
    lastClick = micros();
    while (micros() - error_cue_start < error_cue_dur){
        if ((micros() - lastClick) > 2) { // Changing this value changes the frequency.
            lastClick = micros();
            digitalWrite(SPEAKERS_PIN, generateNoise());
        }
    }
}

/*
##     ##    ###    ##       ##     ## ########
##     ##   ## ##   ##       ##     ## ##
##     ##  ##   ##  ##       ##     ## ##
##     ## ##     ## ##       ##     ## ######
 ##   ##  ######### ##        ##   ##  ##
  ## ##   ##     ## ##         ## ##   ##
   ###    ##     ## ########    ###    ########
*/


// west
bool reward_valve_west_is_closed = true;
unsigned long t_reward_valve_west_open = max_future;

// east
bool reward_valve_east_is_closed = true;
unsigned long t_reward_valve_east_open = max_future;

void reward_valve_controller(){
    // a self terminating digital pin switch
    // flipped by setting deliver_reward to true somewhere in the FSM

    // WEST
    if (reward_valve_west_is_closed == true && deliver_reward_west == true) {

        // send one pulse to pump pin before doing anything
        digitalWrite(REWARD_PUMP_PIN, HIGH);
        digitalWrite(REWARD_PUMP_PIN, LOW); // TODO: one rise?
        
        digitalWrite(REWARD_WEST_VALVE_PIN, HIGH);
        log_code(WATER_VALVE_WEST_ON);
        reward_valve_west_is_closed = false;
        t_reward_valve_west_open = now();

        deliver_reward_west = false;
    }

    if (reward_valve_west_is_closed == false && now() - t_reward_valve_west_open > reward_valve_dur) {
        digitalWrite(REWARD_WEST_VALVE_PIN, LOW);
        log_code(WATER_VALVE_WEST_OFF);
        reward_valve_west_is_closed = true;
    }

    // EAST
    if (reward_valve_east_is_closed == true && deliver_reward_east == true) {

        // send one pulse to pump pin before doing anything
        digitalWrite(REWARD_PUMP_PIN, HIGH);
        digitalWrite(REWARD_PUMP_PIN, LOW); // TODO: one rise?
        
        digitalWrite(REWARD_EAST_VALVE_PIN, HIGH);
        log_code(WATER_VALVE_EAST_ON);
        reward_valve_east_is_closed = false;
        t_reward_valve_east_open = now();

        deliver_reward_east = false;
    }

    if (reward_valve_east_is_closed == false && now() - t_reward_valve_east_open > reward_valve_dur) {
        digitalWrite(REWARD_EAST_VALVE_PIN, LOW);
        log_code(WATER_VALVE_EAST_OFF);
        reward_valve_east_is_closed = true;
    }
}

/*
 
  ######  ##    ## ##    ##  ######  
 ##    ##  ##  ##  ###   ## ##    ## 
 ##         ####   ####  ## ##       
  ######     ##    ## ## ## ##       
       ##    ##    ##  #### ##       
 ##    ##    ##    ##   ### ##    ## 
  ######     ##    ##    ##  ######  
 
*/

bool switch_sync_pin = false;
bool sync_pin_is_on = false;
unsigned long t_last_sync_pin_on = max_future;
unsigned long sync_pulse_dur = 100;

void sync_pin_controller(){
    // switch on
    if (switch_sync_pin == true){
        digitalWrite(CAM_SYNC_PIN, HIGH);
        sync_pin_is_on = true;
        switch_sync_pin = false;
        t_last_sync_pin_on = now();
    }
    // switch off
    if (sync_pin_is_on == true && now() - t_last_sync_pin_on > sync_pulse_dur){
        digitalWrite(CAM_SYNC_PIN, LOW);
        sync_pin_is_on = false;
    }
}

/*
######## ########  ####    ###    ##          ######## ##    ## ########  ########
   ##    ##     ##  ##    ## ##   ##             ##     ##  ##  ##     ## ##
   ##    ##     ##  ##   ##   ##  ##             ##      ####   ##     ## ##
   ##    ########   ##  ##     ## ##             ##       ##    ########  ######
   ##    ##   ##    ##  ######### ##             ##       ##    ##        ##
   ##    ##    ##   ##  ##     ## ##             ##       ##    ##        ##
   ##    ##     ## #### ##     ## ########       ##       ##    ##        ########
*/

int i;

void log_choice(){
    if (is_poking_west == true){
        log_code(CHOICE_WEST_EVENT);
    }
    if (is_poking_east == true){
        log_code(CHOICE_EAST_EVENT);
    }
}

/*
########  ######  ##     ##
##       ##    ## ###   ###
##       ##       #### ####
######    ######  ## ### ##
##             ## ##     ##
##       ##    ## ##     ##
##        ######  ##     ##
*/

void state_entry_common(){
    // common tasks to do at state entry for all states
    last_state = current_state;
    t_state_entry = now();
    log_code(current_state);
}

void finite_state_machine(){
    // the main FSM
    switch (current_state){

        case INI_STATE:
            current_state = ITI_STATE;
            break;

        case TRIAL_AVAILABLE_STATE:

            // state entry
            if (current_state != last_state){
                state_entry_common();
                log_code(TRIAL_AVAILABLE_EVENT);

                trial_available_cue();
            }

            if (true) {
                current_state = CHOICE_STATE;
            }
            break;

        case CHOICE_STATE:
            // state entry
            if (current_state != last_state){
                state_entry_common();
            }

            // choice made 
            if (is_poking_west == true || is_poking_east == true) {

                // turn both LEDS off irrespective of their previous state
                digitalWrite(POKE_EAST_LED, LOW);
                digitalWrite(POKE_WEST_LED, LOW);

                log_code(CHOICE_EVENT);
                log_choice();
                        
            // no choice was made
            if (now() - t_state_entry > choice_dur){
                log_code(CHOICE_MISSED_EVENT);
                log_code(TRIAL_UNSUCCESSFUL_EVENT);

                // cue
                incorrect_choice_cue();
                current_state = ITI_STATE;
                break;
            }
            break;

        case REWARD_STATE:
            // state entry
            if (current_state != last_state){
                state_entry_common();
                reward_cue()

                if (is_poking_west == true){
                    log_code(REWARD_WEST_EVENT);
                    deliver_reward_west = true;
                }
                else{
                    log_code(REWARD_EAST_EVENT);
                    deliver_reward_east = true;
                }
            }

            // exit condition
            if (true) {
                current_state = ITI_STATE;
            }
            break;

        case ITI_STATE:
            // state entry
            if (current_state != last_state){
                state_entry_common();
                this_ITI_dur = (unsigned long) random(ITI_dur_min, ITI_dur_max);
            }

            // exit condition
            if (now() - t_state_entry > this_ITI_dur) {
                current_state = TRIAL_AVAILABLE_STATE;
            }
            break;
    }
}


/*
##     ##    ###    #### ##    ##
###   ###   ## ##    ##  ###   ##
#### ####  ##   ##   ##  ####  ##
## ### ## ##     ##  ##  ## ## ##
##     ## #########  ##  ##  ####
##     ## ##     ##  ##  ##   ###
##     ## ##     ## #### ##    ##
*/

void setup() {
    delay(1000);
    Serial.begin(115200); // main serial communication with computer
    
    // TTL COM w camera
    pinMode(CAM_SYNC_PIN,OUTPUT);

    // ini speakers
    pinMode(SPEAKERS_PIN,OUTPUT);
    tone_controller.begin(SPEAKERS_PIN);

    // LED related
    FastLED.addLeds<WS2812B, LED_PIN, GRB>(leds, NUM_LEDS);
    for (int i = 0; i < NUM_LEDS; i++) {
        led_is_on[i] = false;
        switch_led_on[i] = false;
        led_on_time[i] = max_future;
    }

    lights_off();
    Serial.println("<Arduino is ready to receive commands>");
    delay(1000);
}

void loop() {
    if (run == true){
        // execute state machine(s)
        finite_state_machine();
    }

    // Controllers
    reward_valve_controller();

    // sample sensors
    read_pokes();

    // serial communication with main PC
    getSerialData();
    processSerialData();
    
    // non-blocking cam sync pin
    sync_pin_controller();
}