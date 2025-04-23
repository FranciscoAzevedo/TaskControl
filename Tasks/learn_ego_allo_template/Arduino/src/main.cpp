#include <Arduino.h>
#include <Tone.h>
#include <FastLED.h>

#include <event_codes.h> // <>?
#include "interface.cpp"
#include "pin_map.h"
#include "logging.cpp"
#include <Adafruit_NeoPixel.h>

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

// context related
bool is_ego_context = true; // start ego but immediately flip coin
int this_context_dur = 0;
int current_context_counter = 0;

Color bgColor = Color(255, 0, 0);
Color timeoutColor = Color(255, 255, 255);

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
    // turn the respective port light ON
    if (init_port == north){
        digitalWrite(POKE_NORTH_LED, HIGH);
        log_code(LIGHT_NORTH_CUE_EVENT);
    }
    else {
        digitalWrite(POKE_SOUTH_LED, HIGH);
        log_code(LIGHT_SOUTH_CUE_EVENT);
    }
}

void sound_cue(){     
    tone_controller.play(tone_freq, tone_dur);
}

void reward_cue(){
    tone_controller.play(reward_tone_freq, tone_dur);
}

// /*
// adapted from  https://arduino.stackexchange.com/questions/6715/audio-frequency-white-noise-generation-using-arduino-mini-pro
// */
// #define LFSR_INIT  0xfeedfaceUL
// #define LFSR_MASK  ((unsigned long)( 1UL<<31 | 1UL <<15 | 1UL <<2 | 1UL <<1  ))

// unsigned int generateNoise(){ 
//     // See https://en.wikipedia.org/wiki/Linear_feedback_shift_register#Galois_LFSRs
//     static unsigned long int lfsr = LFSR_INIT;
//     if(lfsr & 1) { lfsr =  (lfsr >>1) ^ LFSR_MASK ; return(1);}
//     else         { lfsr >>= 1;                      return(0);}
// }

// unsigned long error_cue_start = max_future;
// unsigned long error_cue_dur = tone_dur * 1000; // to save instructions - work in micros
// unsigned long lastClick = max_future;

// void incorrect_choice_cue(){
//     // white noise - blocking arduino for tone_dur
//     error_cue_start = micros();
//     lastClick = micros();
//     while (micros() - error_cue_start < error_cue_dur){
//         if ((micros() - lastClick) > 2) { // Changing this value changes the frequency.
//             lastClick = micros();
//             digitalWrite(SPEAKERS_PIN, generateNoise());
//         }
//     }
// }

/*
##     ##    ###    ##       ##     ## ########
##     ##   ## ##   ##       ##     ## ##
##     ##  ##   ##  ##       ##     ## ##
##     ## ##     ## ##       ##     ## ######
 ##   ##  ######### ##        ##   ##  ##
  ## ##   ##     ## ##         ## ##   ##
   ###    ##     ## ########    ###    ########
*/


unsigned long t_reward_valve_open = max_future;
bool reward_valve_west_is_closed = true; // west
bool reward_valve_east_is_closed = true; // east
bool reward_pump_is_closed = true;

void reward_valve_controller(){
    // a self terminating digital pin switch
    // flipped by setting deliver_reward to true somewhere in the FSM

    // WEST
    if (reward_valve_west_is_closed == true && deliver_reward_west == true) {

        // send one pulse to pump pin before doing anything
        t_reward_valve_open = now();

        digitalWrite(REWARD_PUMP_PIN, HIGH);
        digitalWrite(REWARD_WEST_VALVE_PIN, HIGH);

        log_code(WATER_VALVE_WEST_ON);
        reward_pump_is_closed = false;
        reward_valve_west_is_closed = false;
        deliver_reward_west = false;
    }

    if (reward_valve_west_is_closed == false &&  t_reward_valve_open > reward_valve_dur) {
        digitalWrite(REWARD_WEST_VALVE_PIN, LOW);
        log_code(WATER_VALVE_WEST_OFF);
        reward_valve_west_is_closed = true;
    }

    // EAST
    if (reward_valve_east_is_closed == true && deliver_reward_east == true) {

        // send one pulse to pump pin before doing anything
        t_reward_valve_open = now();

        digitalWrite(REWARD_PUMP_PIN, HIGH);
        digitalWrite(REWARD_EAST_VALVE_PIN, HIGH);

        log_code(WATER_VALVE_EAST_ON);
        reward_pump_is_closed = false;
        reward_valve_east_is_closed = false;
        deliver_reward_east = false;
    }

    if (reward_valve_east_is_closed == false && now() - t_reward_valve_open > reward_valve_dur) {
        digitalWrite(REWARD_EAST_VALVE_PIN, LOW);
        log_code(WATER_VALVE_EAST_OFF);
        reward_valve_east_is_closed = true;
    }

    // PUMP
    if(reward_pump_is_closed == false && now() - t_reward_valve_open > reward_pump_dur){
        digitalWrite(REWARD_PUMP_PIN, LOW);
        reward_pump_is_closed == true;
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

//
void SetNeopixelClr(Adafruit_NeoPixel &neopixel, Color c) {
    SetNeopixelClr(neopixel, c, 1);
}

//
void ClearNeopixel(Adafruit_NeoPixel &neopixel) {
    neopixel.clear();
    neopixel.show();
}

// Sample from a truncated exponential distribution
double TruncExpRnd(double x_mean, double x_min, double x_max) {
    double lambda = 1 / x_mean;
    double u = random(0, 1000001) / 1000000.0;
    double F_min = 1 - exp(-lambda * x_min);
    double F_max = 1 - exp(-lambda * x_max);
    double F = F_min + u * (F_max - F_min);
    double sample = -log(1 - F) / lambda;
    return sample;
}

// Draws a value from a uniform distribution
double UniformDist(int minimum, int maximum) {
    return random(minimum, maximum + 1);
}

  // Draws a value from a truncated exponential distribution
double TruncExpDist(int minimum, int mean, int maximum) {
    double e = -double(mean) * log(double(random(1000000) + 1) / double(1000000));
    if (e > maximum || e < minimum) {
        e = TruncExpDist(minimum, mean, maximum);
    }
    return round(e);
}

stimDelay = TruncExpDist(minStimDelay, meanStimDelay, maxStimDelay);

// no intervals can change session to session, declared in interface_variables
unsigned long this_interval = 1500;
unsigned long short_intervals[no_intervals] = {600,1050,1380};
unsigned long long_intervals[no_intervals] = {2400,1950,1620}; // inverse order matters

float p_short_intervals[no_intervals] = {1/no_intervals,1/no_intervals,1/no_intervals};
float p_long_intervals[no_intervals] = {1/no_intervals,1/no_intervals,1/no_intervals};

int i;
float p_cum;

unsigned long get_short_interval(){
    r = random(0,1000) / 1000.0;
    for (int i = 0; i < no_intervals; i++){
        p_cum = 0;
        for (int j = 0; j <= i; j++){
            p_cum += p_short_intervals[j];
        }
        if (r < p_cum){
            return short_intervals[i];
        }
    }
    return -1;
}

unsigned long get_long_interval(){
    r = random(0,1000) / 1000.0;
    for (int i = 0; i < n_intervals; i++){
        p_cum = 0;
        for (int j = 0; j <= i; j++){
            p_cum += p_long_intervals[j];
        }
        if (r < p_cum){
            return long_intervals[i];
        }
    }
    return -1;
}

void set_interval(){
    // init port -> context -> movement sampled

    if (init_port == north){ // no difference between ego and allo on north port

        // implicit mapping of left to long
        if (correct_movement == right){ 
            this_interval = get_short_interval();
            correct_side = east;
        }
        else{
            this_interval = get_long_interval();
            correct_side = west;
        }
    }

    if (init_port = south){ // need context rule to desambiguate

        if (is_ego_context == true){ // egocentric context rule
            // implicit mapping of left to long
            if (correct_movement == right){ 
                this_interval = get_short_interval();
                correct_side = west;
            }
            else{
                this_interval = get_long_interval();
                correct_side = east;
            }
        }
        else{ // allocentric context rule
            // inverted mapping of left to short
            if (correct_movement == right){ 
                this_interval = get_long_interval();
                correct_side = west;
            }
            else{
                this_interval = get_short_interval();
                correct_side = east;
            }
        }
    }
}

void get_trial_type(){
    
    // update correct movement (ego coordinates)
    r = random(0,1000) / 1000.0;
    if (r > 0.5){
        correct_movement = right;
    }
    else {
        correct_movement = left;
    }

    // now is called to update
    set_interval();

    // logging for analysis
    log_ulong("this_interval", this_interval);
    log_int("correct_movement", correct_movement);
    log_int("correct_side", correct_side);
}

void log_choice(){

    if (init_port == north){ // no difference between ego and allo on north port
        if (is_poking_west == true){
            log_code(CHOICE_WEST_EVENT);
            log_code(CHOICE_LONG_EVENT);
        }
        if (is_poking_east == true){
            log_code(CHOICE_EAST_EVENT);
            log_code(CHOICE_SHORT_EVENT);
        }
    }

    if (init_port == south){ // need context rule to desambiguate
        if (is_ego_context == true){
            if (is_poking_west == true){
                log_code(CHOICE_WEST_EVENT);
                log_code(CHOICE_SHORT_EVENT);
            }
            if (is_poking_east == true){
                log_code(CHOICE_EAST_EVENT);
                log_code(CHOICE_LONG_EVENT);
            }
        }
        else{
            if (is_poking_west == true){
                log_code(CHOICE_WEST_EVENT);
                log_code(CHOICE_LONG_EVENT);
            }
            if (is_poking_east == true){
                log_code(CHOICE_EAST_EVENT);
                log_code(CHOICE_SHORT_EVENT);
            }
        }
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

                // evaluate context
                if (current_context_counter == this_context_dur){ // flip it
                    if (is_ego_context == true){
                        is_ego_context == false;
                    }
                    else{
                        is_ego_context == true;
                    }
                    // reset counter and sample new duration
                    current_context_counter = 0;
                    this_context_dur = (unsigned long) random(block_dur_min, block_dur_max);
                }
                else{ // increase counter
                    current_context_counter++;
                }

                // flip a coin for N or S port
                r = random(0,1000) / 1000.0;
                if (r > 0.5){
                    init_port = north;
                }
                else {
                    init_port = south;
                }
                trial_available_cue();
            }

            if (current_state == last_state){
                // the update loop
                if (init_port == north){
                    if (is_poking_north == true){
                        digitalWrite(POKE_NORTH_LED, LOW);
                        log_code(TRIAL_ENTRY_NORTH_EVENT);
                        current_state = TRIAL_ENTRY_STATE;
                        break;
                    }
                }
                else {
                    if (is_poking_south == true){
                        digitalWrite(POKE_SOUTH_LED, LOW);
                        log_code(TRIAL_ENTRY_SOUTH_EVENT);
                        current_state = TRIAL_ENTRY_STATE;
                        break;
                    }
                }
            }
            break;

        case TRIAL_ENTRY_STATE:
            // state entry
            if (current_state != last_state){
                state_entry_common();

                log_code(TRIAL_ENTRY_EVENT);
                log_code(FIRST_TIMING_CUE_EVENT);
                sound_cue(); // first timing cue

                // sync at trial entry
                switch_sync_pin = true;
                sync_pin_controller(); // and call sync controller for enhanced temp prec.

                // determine the type of trial:
                get_trial_type(); // updates correct_movement and correct_side
            }

            // doesn't update, immediately exits

            // exit condition 
            if (true) {
                current_state = PRESENT_INTERVAL_STATE;
            }
            break;

        case PRESENT_INTERVAL_STATE:
            // state entry
            if (current_state != last_state){
                state_entry_common();
            }

            // update
            if (last_state == current_state){
                if (is_poking == true){
                    t_poke_remain = now();
                }

                if (is_poking == false && now()-t_poke_remain > grace_period){ // 50 ms grace period
                    // trial broken
                    log_code(TRIAL_BROKEN_EVENT);
                    log_code(TRIAL_UNSUCCESSFUL_EVENT);
                    incorrect_choice_cue();

                    current_state = TIMEOUT_STATE;
                    break;
                }
            }

            // if fixation is successful, go clock a choice
            if (now() - t_state_entry > this_interval){

                sound_cue();
                log_code(SECOND_TIMING_CUE_EVENT);

                // cue
                if (correct_side == west){
                    go_cue_west();
                }
                if (correct_side == east){
                    go_cue_east();
                }
                current_state = CHOICE_STATE;
                break;
            }
            break;

        case CHOICE_STATE:
            // state entry
            if (current_state != last_state){
                state_entry_common();
            }

            // exit conditions

            // choice made 
            if (is_poking_west == true || is_poking_east == true) {

                // turn both LEDS off irrespective of their previous state
                digitalWrite(POKE_EAST_LED, LOW);
                digitalWrite(POKE_WEST_LED, LOW);

                log_code(CHOICE_EVENT);
                log_choice();

                // correct choice
                if ((correct_side == west && is_poking_west) || (correct_side == east && is_poking_east)){
                    log_code(CHOICE_CORRECT_EVENT);
                    log_code(TRIAL_SUCCESSFUL_EVENT);

                    // succ_trial_counter += 1;
                    // if (correct_side == left){
                    //     left_error_counter = 0;
                    // }

                    // if (correct_side == right){
                    //     right_error_counter = 0;
                    // }

                    current_state = REWARD_STATE;
                    break;
                }

                // incorrect choice
                if ((correct_side == west && is_poking_east) || (correct_side == east && is_poking_west)){
                    log_code(CHOICE_INCORRECT_EVENT);
                    log_code(TRIAL_UNSUCCESSFUL_EVENT);
                    incorrect_choice_cue();

                    // // update counters
                    // if (correct_side == left){
                    //     left_error_counter += 1;
                    //     right_error_counter = 0;
                    // }
                    // if (correct_side == right){
                    //     right_error_counter += 1;
                    //     left_error_counter = 0;
                    // }
                    // if (corr_loop_reset_mode == true){
                    //     succ_trial_counter = 0;
                    // }

                    current_state = TIMEOUT_STATE;
                    break;
                }
            }
                        
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

                if (correct_side == west){
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

        case TIMEOUT_STATE:
            // state entry
            if (current_state != last_state){
                state_entry_common();
            }
            
            // play white noise
            if(current_state == last_state && now()- t_state_entry > tone_dur){
                digitalWrite(SPEAKERS_PIN, !digitalRead(SPEAKERS_PIN)); 
            }

            // exit
            if (now() - t_state_entry > timeout_dur){
                current_state = ITI_STATE;
                break;
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