#include <Arduino.h>
#include <Tone.h>
#include <FastLED.h>

#include <event_codes.h> // <>?
#include "interface.cpp"
// #include "raw_interface.cpp"
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

// int current_state = INI_STATE; // starting at this, aleady declared in interface.cpp
int last_state = -1; // whatever other state
unsigned long max_future = 4294967295; // 2**32 -1
unsigned long t_state_entry = max_future;
unsigned long this_ITI_dur;

// for random
float r;
unsigned long rand_dur;
bool call_block;

int left = 4;
int right = 6;

int choice;
int correct_side;

// laterality related 
int last_correct_side = left;
// int this_correct_side = right;

// fwd declare necessary
bool timing_trial = false;

// bias related
// float bias = 0.5; // exposed in interface_variables.h
int n_choices_left = 1;
int n_choices_right = 1;

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
// Tone tone_controller_left;
// Tone tone_controller_right;

// buzzer
Tone buzz_controller_left;
Tone buzz_controller_right;

// void trial_available_cue(){
//     lights_off();
// }

// void trial_entry_cue(){     
//     buzz_controller_left.play(buzz_center_freq, trial_entry_buzz_dur);
//     buzz_controller_right.play(buzz_center_freq, trial_entry_buzz_dur);
// }

void reward_left_cue(){
    // tone_controller_left.play(go_cue_freq, tone_dur);
    if (lateral_cues == 1){
        buzz_controller_left.play(buzz_center_freq, buzz_dur);
    }
    else{
        if (timing_trial == false){
            buzz_controller_left.play(buzz_center_freq-buzz_freq_sep, buzz_dur);
            buzz_controller_right.play(buzz_center_freq-buzz_freq_sep, buzz_dur);
        }
        else{
            buzz_controller_left.play(buzz_center_freq, buzz_dur);
            buzz_controller_right.play(buzz_center_freq, buzz_dur);
        }
    }
}

void reward_right_cue(){
    // tone_controller_right.play(go_cue_freq, tone_dur);
    if (lateral_cues == 1){
        buzz_controller_right.play(buzz_center_freq, buzz_dur);
    }
    else{
        if (timing_trial == false){
            buzz_controller_left.play(buzz_center_freq+buzz_freq_sep, buzz_dur);
            buzz_controller_right.play(buzz_center_freq+buzz_freq_sep, buzz_dur);
        }
        else{
            buzz_controller_left.play(buzz_center_freq, buzz_dur);
            buzz_controller_right.play(buzz_center_freq, buzz_dur);            
        }
    }
}

void go_cue_left(){
    log_code(GO_CUE_LEFT_EVENT);
    if (left_short == 1){
        log_code(GO_CUE_SHORT_EVENT);
    }
    else{
        log_code(GO_CUE_LONG_EVENT);
    }
    reward_left_cue();
}

void go_cue_right(){
    log_code(GO_CUE_RIGHT_EVENT);
    if (left_short == 1){
        log_code(GO_CUE_LONG_EVENT);
    }
    else{
        log_code(GO_CUE_SHORT_EVENT);
    }
    reward_right_cue();
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
    // lights on
    lights_on();
    // tone_controller_left.play(2000,1000);
    // white noise - blocking arduino for tone_dur
    error_cue_start = micros();
    lastClick = micros();
    while (micros() - error_cue_start < error_cue_dur){
        if ((micros() - lastClick) > 2 ) { // Changing this value changes the frequency.
            lastClick = micros();
            digitalWrite (SPEAKER_LEFT_PIN, generateNoise());
            digitalWrite (SPEAKER_RIGHT_PIN, generateNoise());
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

unsigned long ul2time(float reward_volume, float valve_ul_ms){
    return (unsigned long) reward_volume / valve_ul_ms;
}

// left
bool reward_valve_left_is_closed = true;
// bool deliver_reward_left = false; // already forward declared in interface.cpp
unsigned long t_reward_valve_left_open = max_future;
unsigned long reward_valve_left_dur;

// right
bool reward_valve_right_is_closed = true;
// bool deliver_reward_right = false; // already forward declared in interface.cpp
unsigned long t_reward_valve_right_open = max_future;
unsigned long reward_valve_right_dur;

void reward_valve_controller(){
    // a self terminating digital pin switch
    // flipped by setting deliver_reward to true somewhere in the FSM
    
    // left
    if (reward_valve_left_is_closed == true && deliver_reward_left == true) {
        digitalWrite(REWARD_LEFT_VALVE_PIN, HIGH);
        log_code(REWARD_LEFT_VALVE_ON);
        reward_valve_left_is_closed = false;
        reward_valve_left_dur = ul2time(reward_magnitude, valve_ul_ms_left);
        t_reward_valve_left_open = now();
        deliver_reward_left = false;
        
        // present cue? (this is necessary for keeping the keyboard reward functionality)
        if (present_reward_left_cue == true){
            reward_left_cue();
            present_reward_left_cue = false;
        }
    }

    if (reward_valve_left_is_closed == false && now() - t_reward_valve_left_open > reward_valve_left_dur) {
        digitalWrite(REWARD_LEFT_VALVE_PIN, LOW);
        log_code(REWARD_LEFT_VALVE_OFF);
        reward_valve_left_is_closed = true;
    }

    // right
    if (reward_valve_right_is_closed == true && deliver_reward_right == true) {
        digitalWrite(REWARD_RIGHT_VALVE_PIN, HIGH);
        log_code(REWARD_RIGHT_VALVE_ON);
        reward_valve_right_is_closed = false;
        reward_valve_right_dur = ul2time(reward_magnitude, valve_ul_ms_right);
        t_reward_valve_right_open = now();
        deliver_reward_right = false;
        
        // present cue? (this is necessary for keeping the keyboard reward functionality)
        if (present_reward_right_cue == true){
            reward_right_cue();
            present_reward_right_cue = false;
        }
    }

    if (reward_valve_right_is_closed == false && now() - t_reward_valve_right_open > reward_valve_right_dur) {
        digitalWrite(REWARD_RIGHT_VALVE_PIN, LOW);
        log_code(REWARD_RIGHT_VALVE_OFF);
        reward_valve_right_is_closed = true;
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
        digitalWrite(LC_SYNC_PIN, HIGH);
        sync_pin_is_on = true;
        switch_sync_pin = false;
        t_last_sync_pin_on = now();
    }
    // switch off
    if (sync_pin_is_on == true && now() - t_last_sync_pin_on > sync_pulse_dur){
        digitalWrite(CAM_SYNC_PIN, LOW);
        digitalWrite(LC_SYNC_PIN, LOW);
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

// bool correction_loops = true;
// int correction_loops = 1;
bool in_corr_loop = false;
int left_error_counter = 0;
int right_error_counter = 0;
int succ_trial_counter = 0;
bool corr_loop_reset_mode = true;
unsigned long this_interval = 1500;

/*
resetting mode:
within correction loop, any mistake restarts the counter from the beginning
no resetting: intermediate mistakes allowed, corr loop is exited after 3 correct choices
*/

unsigned long get_short_interval(){
    return random(600, 1500-gap);
}

unsigned long get_long_interval(){
    return random(1500+gap, 2400);
}

void set_interval(){
    if (correct_side == right){
        if (left_short == 0){ // right short
            this_interval = get_short_interval();
        }
        else{
            this_interval = get_long_interval();
        }
    }

    if (correct_side == left){
        if (left_short == 1){
            this_interval = get_short_interval();
        }
        else {
            this_interval = get_long_interval();
        }
    }
}

void get_trial_type(){
    if (correction_loops == 1){
        // determine if enter corr loop
        if (in_corr_loop == false && (left_error_counter >= corr_loop_entry || right_error_counter >= corr_loop_entry)){
            in_corr_loop = true;
            timing_trial = false;
        }
        
        // determine if exit corr loop
        if (in_corr_loop == true && succ_trial_counter >= corr_loop_exit){
            in_corr_loop = false;
        }
    }
    
    // if not in corr loop, randomly choose new correct side
    if (in_corr_loop == false){ 
        // update correct side
        r = random(0,1000) / 1000.0;
        if (r > 0.5){
            correct_side = right;
        }
        else {
            correct_side = left;
        }
        // update timing trial
        r = random(0,1000) / 1000.0;
        if (r < p_timing_trial){
            timing_trial = true;
        }
        else {
            timing_trial = false;
        }
    }
    else{
        // if in corr loop, this is not updated
    }

    // now is always called to update even in corr loop
    set_interval();

    log_ulong("this_interval", this_interval);
    log_int("correct_side", correct_side);
    log_int("in_corr_loop", (int) in_corr_loop);
    log_int("timing_trial", (int) timing_trial);
}
              

void log_choice(){
    if (is_reaching_left == true){
        log_code(CHOICE_LEFT_EVENT);
        n_choices_left++;
        if (left_short == 1){
            log_code(CHOICE_SHORT_EVENT);
        }
        else{
            log_code(CHOICE_LONG_EVENT);
        }
    }
    if (is_reaching_right == true){
        log_code(CHOICE_RIGHT_EVENT);
        n_choices_right++;
        if (left_short == 1){
            log_code(CHOICE_LONG_EVENT);
        }
        else{
            log_code(CHOICE_SHORT_EVENT);
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

void finite_state_machine() {
    // the main FSM
    switch (current_state) {

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

            if (current_state == last_state){
                // the update loop
                if (trial_autostart == 0){
                    if (digitalRead(TRIAL_INIT_PIN) == true && now() - t_last_reach_off > reach_block_dur && is_reaching == false){
                        current_state = TRIAL_ENTRY_STATE;
                        break;
                    }
                }
                else{
                    current_state = TRIAL_ENTRY_STATE;
                    break;
                }
            }
            break;

        case TRIAL_ENTRY_STATE:
            // state entry
            if (current_state != last_state){
                state_entry_common();
                log_code(TRIAL_ENTRY_EVENT);
                trial_entry_cue(); // which is first timing cue

                // sync at trial entry
                switch_sync_pin = true;
                sync_pin_controller(); // and call sync controller for enhanced temp prec.

                // bias related
                // update_bias();
                
                // if (correction_loops == 0){
                //     update_p_reward();
                // }

                // log bias
                // log_float("bias", bias);

                // determine the type of trial:
                get_trial_type(); // updates this_correct_side

            }

            // update
            if (last_state == current_state){

            }
            
            // exit condition 
            // trial autostart
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
                if (is_reaching == true){
                    // premature choice
                    log_code(CHOICE_EVENT);
                    log_code(PREMATURE_CHOICE_EVENT);
                    log_code(TRIAL_UNSUCCESSFUL_EVENT);
                    incorrect_choice_cue();
                    log_choice();
                    current_state = TIMEOUT_STATE;
                    break;
                }
            }

            if (now() - t_state_entry > this_interval){
                // cue
                if (correct_side == left){
                    go_cue_left();
                }
                if (correct_side == right){
                    go_cue_right();
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

            // update
            // if (last_state == current_state){
            // }

            // exit conditions

            // choice was made
            if (is_reaching == true && now() - t_last_reach_on > min_grasp_dur) {
                log_code(CHOICE_EVENT);
                log_choice();

                // correct choice
                if ((correct_side == left && is_reaching_left) || (correct_side == right && is_reaching_right)){
                    log_code(CHOICE_CORRECT_EVENT);
                    log_code(TRIAL_SUCCESSFUL_EVENT);

                    succ_trial_counter += 1;
                    if (correct_side == left){
                        left_error_counter = 0;
                    }

                    if (correct_side == right){
                        right_error_counter = 0;
                    }
                    current_state = REWARD_STATE;
                    break;
                }

                // incorrect choice
                if ((correct_side == left && is_reaching_right) || (correct_side == right && is_reaching_left)){
                    log_code(CHOICE_INCORRECT_EVENT);
                    log_code(TRIAL_UNSUCCESSFUL_EVENT);
                    incorrect_choice_cue();

                    // update counters
                    if (correct_side == left){
                        left_error_counter += 1;
                        right_error_counter = 0;
                    }
                    if (correct_side == right){
                        right_error_counter += 1;
                        left_error_counter = 0;
                    }
                    if (corr_loop_reset_mode == true){
                        succ_trial_counter = 0;
                    }

                    current_state = TIMEOUT_STATE;
                    break;
                }
            }
                        
            // no choice was made
            if (now() - t_state_entry > choice_dur){
                log_code(CHOICE_MISSED_EVENT);
                log_code(TRIAL_UNSUCCESSFUL_EVENT);

                // cue
                // incorrect_choice_cue();
                current_state = ITI_STATE;
                break;
            }

            break;

        case REWARD_STATE:
            // state entry
            if (current_state != last_state){
                state_entry_common();
                if (correct_side == left){
                    log_code(REWARD_LEFT_EVENT);
                    deliver_reward_left = true;
                }
                else{
                    log_code(REWARD_RIGHT_EVENT);
                    deliver_reward_right = true;
                }
            }

            // exit condition
            if (true) {
                // transit to ITI after certain time
                current_state = ITI_STATE;
            }
            break;

        case TIMEOUT_STATE:
            // state entry
            if (current_state != last_state){
                state_entry_common();
            }

            // update
            if (last_state == current_state){

            }

            // exit condition
            if (now() - t_state_entry > timeout_dur) {
                current_state = ITI_STATE;
            }
            break;

        case ITI_STATE:
            // state entry
            if (current_state != last_state){
                state_entry_common();
                this_ITI_dur = (unsigned long) random(ITI_dur_min, ITI_dur_max);
                rand_dur = random(1000,2000);
                call_block = false;
            }

            // update
            if (last_state == current_state){
                // turn lights on after random duration
                if (now() - t_state_entry > rand_dur && call_block == false) {
                    lights_on();
                    call_block = true;
                }
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

    // TTL com with firmata
    pinMode(REACH_LEFT_PIN, INPUT);
    pinMode(REACH_RIGHT_PIN, INPUT);
    
    // TTL COM w camera
    pinMode(CAM_SYNC_PIN,OUTPUT);
    pinMode(LC_SYNC_PIN,OUTPUT);

    pinMode(SPEAKER_LEFT_PIN, OUTPUT);
    pinMode(SPEAKER_RIGHT_PIN, OUTPUT);

    // ini speakers
    // tone_controller_left.begin(SPEAKER_LEFT_PIN);
    // tone_controller_right.begin(SPEAKER_RIGHT_PIN);

    // ini buzzers
    buzz_controller_left.begin(BUZZER_LEFT_PIN);
    buzz_controller_right.begin(BUZZER_RIGHT_PIN);

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