#include <Arduino.h>
#include <Tone.h>
#include <Adafruit_NeoPixel.h>

#include <event_codes.h> // <>?
#include "colors.h"
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

// Color class
Color::Color()
{
    _r = 0;
    _g = 0;
    _b = 0;
}
Color::Color(int r, int g, int b)
{
    _r = r;
    _g = g;
    _b = b;
}
int Color::getR() const{ return _r;}
int Color::getG() const{ return _g;}
int Color::getB() const{ return _b;}

int last_state = -1; // whatever other state
unsigned long max_future = 4294967295; // 2**32 -1
unsigned long t_state_entry = max_future;
long trial_init_time = 0; // to avoid first trial issues

// Parameters from Thiago Gouvea's eLife paper
unsigned long tone_dur = 150;
unsigned long tone_freq = 7500;
unsigned long reward_tone_freq = 1750;
unsigned long timing_boundary = 1500;

// Parameters for current pumps and pokes
unsigned long reward_valve_dur = 2000; // more than enough for pump
unsigned long reward_pump_toggle_dur = 3; // ms
int targetToggles = 70; // Total number of toggles to perform , double of pump steps
unsigned long grace_period = 100; // ms to avoid poke fluctuations

// speaker
Tone tone_control_east;
Tone tone_control_west;
unsigned long error_cue_start = max_future;
unsigned long error_cue_dur = tone_dur * 1000; // to save instructions - work in micros

//  named variables that are easier to compare in numbers than strings
int north = 8;
int south = 2;
int west = 4;
int east = 6;

// ego centric coords
int left = 7;
int right = 9;

// non-initialized variables
int i;
float r; // for random processes
long this_ITI_dur;
unsigned long t_poke_remain;
bool prev_trial_broken = false; // to avoid resampling trial type if broken fixation in previous trial
bool spkrState;
bool pumpState;
int choice;
int correct_side;
int correct_movement;
int init_port;
bool is_ego_context;

int timeout_flag = 0; // 0 = no timeout, 1 = timeout

// context and port related
int this_context_dur = 0;
int current_context_counter = 0;

int this_init_block_dur = 0;
int current_init_block_counter = 0;

// corr loops TODO: adapt for stim specific instead of side
bool in_corr_loop = false;
int west_error_counter = 0;
int east_error_counter = 0;
int succ_trial_counter = 0;
int trial_counter = 0;

/*
.########..####.##....##.....######...#######..##....##.########.####..######..
.##.....##..##..###...##....##....##.##.....##.###...##.##........##..##....##.
.##.....##..##..####..##....##.......##.....##.####..##.##........##..##.......
.########...##..##.##.##....##.......##.....##.##.##.##.######....##..##...####
.##.........##..##..####....##.......##.....##.##..####.##........##..##....##.
.##.........##..##...###....##....##.##.....##.##...###.##........##..##....##.
.##........####.##....##.....######...#######..##....##.##.......####..######..
*/

void PinInit(){

    // water
    digitalWrite(REWARD_WEST_VALVE_PIN, LOW);
    digitalWrite(REWARD_EAST_VALVE_PIN, LOW);
    digitalWrite(REWARD_PUMP_PIN, LOW);

    // cam
    digitalWrite(CAM_SYNC_PIN, LOW); // turn off camera sync pin

    // speakers
    digitalWrite(SPEAKER_WEST_PIN, 1); // turn off west speaker
    digitalWrite(SPEAKER_EAST_PIN, 1); // turn off west speaker

    // lights
    digitalWrite(BCKGND_LIGHTS_PIN, HIGH); // turn on background lights

    // pokes
    for (i = 0; i < NUM_POKES; i++){
        digitalWrite(POKES_PINS[i], LOW); 
    }
}


/*
.########...#######..##....##.########..######.
.##.....##.##.....##.##...##..##.......##....##
.##.....##.##.....##.##..##...##.......##......
.########..##.....##.#####....######....######.
.##........##.....##.##..##...##.............##
.##........##.....##.##...##..##.......##....##
.##.........#######..##....##.########..######.
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

void read_pokes(){
    // north
    poke_north = digitalRead(POKE_NORTH_PIN);
    if (is_poking_north == false && poke_north == true){
        log_code(POKE_NORTH_IN);
        is_poking_north = true;
    }

    if (is_poking_north == true && poke_north == false){
        log_code(POKE_NORTH_OUT);
        is_poking_north = false;
    }

    // south
    poke_south = digitalRead(POKE_SOUTH_PIN);
    if (is_poking_south == false && poke_south == true){
        log_code(POKE_SOUTH_IN);
        is_poking_south = true;
    }

    if (is_poking_south == true && poke_south == false){
        log_code(POKE_SOUTH_OUT);
        is_poking_south = false;
    }

    // west
    poke_west = digitalRead(POKE_WEST_PIN);
    if (is_poking_west == false && poke_west == true){
        log_code(POKE_WEST_IN);
        is_poking_west = true;
    }

    if (is_poking_west == true && poke_west == false){
        log_code(POKE_WEST_OUT);
        is_poking_west = false;
    }

    // east
    poke_east = digitalRead(POKE_EAST_PIN);
    if (is_poking_east == false && poke_east == true){
        log_code(POKE_EAST_IN);
        is_poking_east = true;
    }

    if (is_poking_east == true && poke_east == false){
        log_code(POKE_EAST_OUT);
        is_poking_east = false;
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

// LED related
#define NUM_BCKGND_PIXELS 64
#define NUM_LED_PIXELS 16
#define NUM_POKES 4

Adafruit_NeoPixel bgNeopixel;
Adafruit_NeoPixel pokesNeopixel[NUM_POKES];

Color redColor = Color(255, 0, 0); // red
Color whiteColor = Color(255, 255, 255); // white
Color egoColor = Color(255, 255, 0); // yellow
Color alloColor = Color(0,0,255); // blue

float offBrightness = 0.0;
float dimBrightness = 0.1;
float halfBrightness = 0.5;
float fullBrightness = 1.0;

void SetNeopixelClr(Adafruit_NeoPixel &neopixel, Color c, float b) {
    neopixel.clear();
    for (unsigned int i = 0; i < neopixel.numPixels(); i++) {
      neopixel.setPixelColor(i, c.getR() * b, c.getG() * b, c.getB() * b);
    }
    neopixel.show();
}

void ClearNeopixel(Adafruit_NeoPixel &neopixel) {
    neopixel.clear();
    neopixel.show();
}

/*
 ######  ##     ## ########  ######
##    ## ##     ## ##       ##    ##
##       ##     ## ##       ##
##       ##     ## ######    ######
##       ##     ## ##             ##
##    ## ##     ## ##       ##    ##
 ######   #######  ########  ######
*/

void trial_available_cue(){
    // turn the respective port light ON
    if (init_port == north){
        if(is_ego_context == true){
            SetNeopixelClr(pokesNeopixel[0], egoColor, fullBrightness);
            log_code(LIGHT_NORTH_CUE_EVENT);
        }
        else {
            SetNeopixelClr(pokesNeopixel[0], alloColor, fullBrightness);
            log_code(LIGHT_NORTH_CUE_EVENT);
        }

    }
    else { // south
        if(is_ego_context == true){
            SetNeopixelClr(pokesNeopixel[1], egoColor, fullBrightness);
            log_code(LIGHT_SOUTH_CUE_EVENT);
        }
        else {
            SetNeopixelClr(pokesNeopixel[1], alloColor, fullBrightness);
            log_code(LIGHT_SOUTH_CUE_EVENT);
        }
    }
}

void go_cue_west(){
    SetNeopixelClr(pokesNeopixel[2], whiteColor, fullBrightness);
    log_code(LIGHT_WEST_CUE_EVENT);
}

void go_cue_east(){
    SetNeopixelClr(pokesNeopixel[3], whiteColor, fullBrightness);
    log_code(LIGHT_EAST_CUE_EVENT);
}

void sound_cue(){
    tone_control_west.play(tone_freq, tone_dur);
    tone_control_east.play(tone_freq, tone_dur);
}

void reward_cue(){
    tone_control_west.play(reward_tone_freq, tone_dur);
    tone_control_east.play(reward_tone_freq, tone_dur);
}

void incorrect_choice_cue(){
    error_cue_start = micros();
    while (micros() - error_cue_start < error_cue_dur){
        spkrState = random(0,2);
        digitalWrite(SPEAKER_WEST_PIN, spkrState);
        digitalWrite(SPEAKER_EAST_PIN, spkrState);
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

unsigned long t_reward_valve_open = max_future;
bool reward_valve_west_is_closed = true; // west
bool reward_valve_east_is_closed = true; // east

void reward_valve_controller(){
    // a self terminating digital pin switch
    // flipped by setting deliver_reward to true somewhere in the FSM

    // WEST
    if (reward_valve_west_is_closed == true && deliver_reward_west == true) {

        t_reward_valve_open = now();
        digitalWrite(REWARD_WEST_VALVE_PIN, HIGH);
        log_code(WATER_WEST_VALVE_ON);
        reward_valve_west_is_closed = false;
        deliver_reward_west = false;
    }

    if (reward_valve_west_is_closed == false &&  now() - t_reward_valve_open > reward_valve_dur) {
        digitalWrite(REWARD_WEST_VALVE_PIN, LOW);
        log_code(WATER_WEST_VALVE_OFF);
        reward_valve_west_is_closed = true;
    }

    // EAST
    if (reward_valve_east_is_closed == true && deliver_reward_east == true) {

        t_reward_valve_open = now();
        digitalWrite(REWARD_EAST_VALVE_PIN, HIGH);
        log_code(WATER_EAST_VALVE_ON);
        reward_valve_east_is_closed = false;
        deliver_reward_east = false;
    }

    if (reward_valve_east_is_closed == false && now() - t_reward_valve_open > reward_valve_dur) {
        digitalWrite(REWARD_EAST_VALVE_PIN, LOW);
        log_code(WATER_EAST_VALVE_OFF);
        reward_valve_east_is_closed = true;
    }
}

/*
.########..##.....##.##.....##.########.
.##.....##.##.....##.###...###.##.....##
.##.....##.##.....##.####.####.##.....##
.########..##.....##.##.###.##.########.
.##........##.....##.##.....##.##.......
.##........##.....##.##.....##.##.......
.##.........#######..##.....##.##.......
*/

int toggleCount = 0;                // Tracks the number of toggles  

void pump_controller() {
    if (togglingActive) {
        unsigned long currentMillis = millis();
        if (currentMillis - previousMillis >= reward_pump_toggle_dur) {

            previousMillis = currentMillis;
            pumpState = digitalRead(REWARD_PUMP_PIN); 
            digitalWrite(REWARD_PUMP_PIN, !pumpState); // Toggle the pin state
            toggleCount++;

            if (toggleCount >= targetToggles) {

                togglingActive = false; // Stop toggling after the desired number of cycles
                toggleCount = 0;
                log_code(WATER_PUMP_OFF);
            }
        }
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

// no. of intervals can change session to session, declared in interface_variables
unsigned long this_interval;

const int max_no_intervals = 3; // max no. of intervals

// allocate with max_no_intervals, limit with no_intervals
unsigned long short_intervals[max_no_intervals] = {600,1050,1380};
unsigned long long_intervals[max_no_intervals] = {2400,1950,1620}; // inverse order matters

float p_short_intervals[max_no_intervals] = {1.0/no_intervals,1.0/no_intervals,1.0/no_intervals};
float p_long_intervals[max_no_intervals] = {1.0/no_intervals,1.0/no_intervals,1.0/no_intervals};

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
    for (int i = 0; i < no_intervals; i++){
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

void  set_interval(){
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

    if (init_port == south){ // need context rule to desambiguate

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

    if (use_correction_loops == 1){

        // determine if enter corr loop
        if (in_corr_loop == false && (west_error_counter >= corr_loop_entry || east_error_counter >= corr_loop_entry)){
            in_corr_loop = true;
            log_int("in_corr_loop", (int) in_corr_loop);
        }
        // determine if exit corr loop
        else if (in_corr_loop == true && succ_trial_counter >= corr_loop_exit){
            in_corr_loop = false;
            log_int("in_corr_loop", (int) in_corr_loop);
        }
    }

    if (in_corr_loop == false && prev_trial_broken == false){ // update

        // update correct movement (ego coordinates)
        r = random(0,1000) / 1000.0;
        if (r > 0.5){
            correct_movement = right;
        }
        else {
            correct_movement = left;
        }

        set_interval();
    }
    else{
        // if in corr loop or resample broken fixation, trial type is not updated
        set_interval(); // need to reevaluate anyway due to changes in init_port and context
    }

    // logging for analysis
    trial_counter++;
    log_int("trial_counter", trial_counter);
    log_ulong("this_interval", this_interval);
    log_int("correct_movement", correct_movement);
    log_int("correct_side", correct_side);
    log_int("is_ego_context", (int) is_ego_context);
}

void log_choice(){

    if (init_port == north){ // no difference between ego and allo on north port
        if (is_poking_west == true){
            log_code(CHOICE_LEFT_EVENT);
            log_code(CHOICE_WEST_EVENT);
            log_code(CHOICE_LONG_EVENT);
        }
        if (is_poking_east == true){
            log_code(CHOICE_RIGHT_EVENT);
            log_code(CHOICE_EAST_EVENT);
            log_code(CHOICE_SHORT_EVENT);
        }
    }

    if (init_port == south){ // need context rule to desambiguate
        if (is_ego_context == true){
            if (is_poking_west == true){
                log_code(CHOICE_RIGHT_EVENT);
                log_code(CHOICE_WEST_EVENT);
                log_code(CHOICE_SHORT_EVENT);
            }
            if (is_poking_east == true){
                log_code(CHOICE_LEFT_EVENT);
                log_code(CHOICE_EAST_EVENT);
                log_code(CHOICE_LONG_EVENT);
            }
        }
        else{
            if (is_poking_west == true){
                log_code(CHOICE_RIGHT_EVENT);
                log_code(CHOICE_WEST_EVENT);
                log_code(CHOICE_LONG_EVENT);
            }
            if (is_poking_east == true){
                log_code(CHOICE_LEFT_EVENT);
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
                        is_ego_context = false;
                    }
                    else{
                        is_ego_context = true;
                    }
                    // reset counter and sample new duration
                    log_int("is_ego_context", (int) is_ego_context);
                    current_context_counter = 0;
                    this_context_dur = (unsigned long) random(block_dur_min, block_dur_max);
                    log_int("this_context_dur", this_context_dur);
                }
                else{ // increase counter
                    current_context_counter++;
                }

                // evaluate port
                if (init_port_blocks == 0){ // no blocks
                    // flip a coin for N or S port
                    r = random(0,1000) / 1000.0;
                    if (r > 0.5){
                        init_port = north;
                    }
                    else {
                        init_port = south;
                    }
                    log_int("init_port", init_port);
                }
                else{ // block initation port
                    if (current_init_block_counter == this_init_block_dur){ // flip it
                        if (init_port == north){
                            init_port = south;
                        }
                        else{
                            init_port = north;
                        }
                        // reset counter and sample new duration
                        log_int("init_port", init_port);
                        current_init_block_counter = 0;
                        this_init_block_dur = (unsigned long) random(port_dur_min, port_dur_max);
                        log_int("this_init_block_dur", this_init_block_dur);
                    }
                    else{ // increase counter
                        current_init_block_counter++;
                    }
                }
                
                trial_available_cue();
            }

            if (current_state == last_state){
                // the update loop
                if (init_port == north){
                    if (is_poking_north == true){
                        if (is_ego_context == true){
                            SetNeopixelClr(pokesNeopixel[0], egoColor, dimBrightness);
                        }
                        else {
                            SetNeopixelClr(pokesNeopixel[0], alloColor, dimBrightness);
                        }

                        log_code(TRIAL_ENTRY_NORTH_EVENT);
                        current_state = TRIAL_ENTRY_STATE;
                        break;
                    }
                }
                else {
                    if (is_poking_south == true){
                        if (is_ego_context == true){
                            SetNeopixelClr(pokesNeopixel[1], egoColor, dimBrightness);
                        }
                        else {
                            SetNeopixelClr(pokesNeopixel[1], alloColor, dimBrightness);
                        }

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
                
                trial_init_time = now();

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
                t_poke_remain = now();
            }

            // update
            if (last_state == current_state){
                if (is_poking == true){
                    t_poke_remain = now();
                }

                if (is_poking == false && now()-t_poke_remain > grace_period){

                    // trial broken
                    ClearNeopixel(pokesNeopixel[0]);
                    ClearNeopixel(pokesNeopixel[1]);

                    prev_trial_broken = true; // set flag to resample trial type
                    
                    log_code(INIT_POKEOUT_EVENT);
                    log_code(BROKEN_FIXATION_EVENT);
                    log_code(TRIAL_UNSUCCESSFUL_EVENT);
                    incorrect_choice_cue();
                    
                    timeout_flag = 1; // set timeout delay to 1, so it goes to timeout state
                    current_state = ITI_STATE;
                    break;
                }
            }

            // if fixation is successful, go clock a choice
            if (now() - t_state_entry > this_interval){

                // in case the animal pokes out within grace period but is still a valid fixation
                if (is_poking == false){
                    log_code(INIT_POKEOUT_EVENT);
                }

                ClearNeopixel(pokesNeopixel[0]);
                ClearNeopixel(pokesNeopixel[1]);

                prev_trial_broken = false; // reset flag to sample new trial type

                sound_cue();
                log_code(SECOND_TIMING_CUE_EVENT);

                // flip a coin for cued trial
                r = random(0,1000) / 1000.0;
                if (r < p_cued){
                    log_int("cued_trial", (int) true);

                    if (correct_side == west){
                        go_cue_west();
                    }
                    else if (correct_side == east){
                        go_cue_east();
                    }         
                }
                else { // uncued
                    log_int("cued_trial", (int) false);
                    go_cue_west();
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

                // udpate leds
                ClearNeopixel(pokesNeopixel[2]);
                ClearNeopixel(pokesNeopixel[3]);

                // correct choices
                if ((is_poking_west == true && correct_side == west) || (is_poking_east == true && correct_side == east)) {
                    
                    // update CORR LOOP counters
                    // succ_trial_counter += 1;
                    // if (correct_side == west){
                    //     west_error_counter = 0;
                    // }

                    // if (correct_side == east){
                    //     east_error_counter = 0;
                    // }

                    reward_cue();
                    log_code(TRIAL_SUCCESSFUL_EVENT);
                    log_code(CHOICE_CORRECT_EVENT);
                    log_code(CHOICE_EVENT);
                    log_choice();
                    current_state = REWARD_STATE;
                }

                // incorrect choices
                if ((correct_side == west && is_poking_east) || (correct_side == east && is_poking_west)){
                    log_code(CHOICE_INCORRECT_EVENT);
                    log_code(TRIAL_UNSUCCESSFUL_EVENT);

                    // update CORR LOOP counters
                    // if (correct_side == west){
                    //     west_error_counter += 1;
                    //     east_error_counter = 0;
                    // }
                    // if (correct_side == east){
                    //     east_error_counter += 1;
                    //     west_error_counter = 0;
                    // }
                    // log_int("west_error_counter", west_error_counter);
                    // log_int("east_error_counter", east_error_counter);
                    // succ_trial_counter = 0;
                    
                    // cue
                    incorrect_choice_cue();
                    current_state = TIMEOUT_STATE;
                    break;
                }
            }
                        
            // no choice was made
            if (now() - t_state_entry > choice_dur){
                // udpate leds
                ClearNeopixel(pokesNeopixel[2]);
                ClearNeopixel(pokesNeopixel[3]);

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
                reward_cue();
                
                // valves
                if (is_poking_west == true){
                    log_code(REWARD_WEST_EVENT);
                    deliver_reward_west = true;
                }
                else{
                    log_code(REWARD_EAST_EVENT);
                    deliver_reward_east = true;
                }

                // pump
                previousMillis = millis();
                togglingActive = true;
                log_code(WATER_PUMP_ON);
            }

            // exit condition
            if (true) {
                current_state = ITI_STATE;
            }
            break;

        case ITI_STATE:

            // speakers
            digitalWrite(SPEAKER_WEST_PIN, 1); // turn off west speaker
            digitalWrite(SPEAKER_EAST_PIN, 1); // turn off east speaker

            // state entry
            if (current_state != last_state){
                state_entry_common();
                
                this_ITI_dur = ITOI_dur-(now()-trial_init_time); // extra duration

                // safeguard for mistakes
                if (this_ITI_dur < 0){
                    log_msg("ITI is negative, setting to 5000 ms");
                    this_ITI_dur = 5000;
                }
                
                log_int("iti_dur", this_ITI_dur);
            }

            // exit condition
            if (timeout_flag == 1){
                current_state = TIMEOUT_STATE;
            }
            else if (now() - t_state_entry > this_ITI_dur) {
                current_state = TRIAL_AVAILABLE_STATE;
            }
            break;

        case TIMEOUT_STATE:
            // state entry
            if (current_state != last_state){
                state_entry_common();
            }

            // exit
            if (now() - t_state_entry > timeout_dur){
                current_state = TRIAL_AVAILABLE_STATE;
                break;
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
    pinMode(SPEAKER_WEST_PIN,OUTPUT);
    tone_control_west.begin(SPEAKER_WEST_PIN);
    pinMode(SPEAKER_EAST_PIN,OUTPUT);
    tone_control_east.begin(SPEAKER_EAST_PIN);

    // ini valves 
    pinMode(REWARD_WEST_VALVE_PIN,OUTPUT);
    pinMode(REWARD_EAST_VALVE_PIN,OUTPUT);
    pinMode(REWARD_PUMP_PIN,OUTPUT);

    // ini pokes
    for (int i = 0; i < NUM_POKES; i++){
        pinMode(POKES_PINS[i],INPUT);
    }

    // ini BGND LEDs 
    pinMode(BCKGND_LIGHTS_PIN,OUTPUT);
    bgNeopixel = Adafruit_NeoPixel(NUM_BCKGND_PIXELS,BCKGND_LIGHTS_PIN,NEO_GRB + NEO_KHZ800);
    bgNeopixel.begin();
    SetNeopixelClr(bgNeopixel, redColor, fullBrightness); // half brightness red background
    bgNeopixel.show();

    // ini POKE LEDs 
    for (int i = 0; i < NUM_POKES; i++){
        pinMode(POKES_LED_PINS[i],OUTPUT);
        pokesNeopixel[i] = Adafruit_NeoPixel(NUM_LED_PIXELS, POKES_LED_PINS[i], NEO_GRB + NEO_KHZ800);
        pokesNeopixel[i].begin();
        ClearNeopixel(pokesNeopixel[i]); // clear poke
        pokesNeopixel[i].show(); // init as off
    }

    PinInit(); // pin initialization

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
    pump_controller();

    // sample sensors
    read_pokes();

    // serial communication with main PC
    getSerialData();
    processSerialData();
    
    // non-blocking cam sync pin
    sync_pin_controller();
}