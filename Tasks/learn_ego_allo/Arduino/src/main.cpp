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
int targetToggles = 71; // Total number of toggles to perform , double of pump steps
unsigned long grace_period = 200; // ms to avoid poke fluctuations

// speaker
Tone tone_control_east;
Tone tone_control_west;
unsigned long error_cue_start = max_future;
unsigned long error_cue_dur = tone_dur * 1000; // to save instructions - work in micros
bool trigger_punish_tone = false; // whether punish tone is active or not
bool punish_tone_ON = false; // whether the tone is playing or not

//  named variables that are easier to compare in numbers than strings
int north = 8;
int south = 2;
int west = 4;
int east = 6;

// ego centric coords
int left = 7;
int right = 9;

// misc non-initialized variables
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
int init_port = 8;
unsigned long this_interval; // no. of intervals changes session to session, declared in interface_variables
float p_cum;

int timeout_flag = 0; // 0 = no timeout, 1 = timeout
bool jittering = false; // whether the animal is jittering or not
bool init_pokeout_logged = false; // flag to log INIT_POKEOUT_EVENT only once per trial

// context and port related
int this_context_dur = 0;
bool is_ego_context = random(0, 2) & 1; // this hasnt been random...
int current_context_counter = 0;

int this_init_block_dur = 0;
int current_init_block_counter = 0;

// timing related
const int max_no_intervals = 3; // max no. of intervals
float p_short_intervals[max_no_intervals] = {0,0,0}; // probabilities for short intervals
float p_long_intervals[max_no_intervals] = {0,0,0}; // probabilities for long intervals

// allocate with max_no_intervals, limit with no_intervals defined in interface_variables
unsigned long short_intervals[max_no_intervals] = {600,1050,1380};
unsigned long long_intervals[max_no_intervals] = {2400,1950,1620}; // inverse order matters

// corr loops for stimulus- and port- specific
bool in_corr_loop = false;
int corr_loop_port_idx = -1;
bool is_short_corr_loop = true; // whether the current corr loop is short or long stim

int idx = 0;
int short_north_error_counter = 0;
int short_south_error_counter = 0;
int long_north_error_counter = 0;
int long_south_error_counter = 0;

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

long last_init_port_entry = 0; // to track the last port entry time

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

    if (is_poking_north == true || is_poking_south == true){
        last_init_port_entry = now();
    }
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

// ego is Left/Right () pattern
void SetEgoNeopixelClr(Adafruit_NeoPixel &neopixel, Color c, float b) {
    neopixel.clear();
    for (int i = 0; i < neopixel.numPixels(); i++) {
        if ((i > neopixel.numPixels() / 2 + 2 && i < neopixel.numPixels() - 1) || (i > 2 && i < (neopixel.numPixels() / 2 -1))) { // left and right
            neopixel.setPixelColor(i, c.getR() * b, c.getG() * b, c.getB() * b);
        }
    }
    neopixel.show();
}

// allo is Up/Down () pattern 
void SetAlloNeopixelClr(Adafruit_NeoPixel &neopixel, Color c, float b) {
    neopixel.clear();
    for (int i = 0; i < neopixel.numPixels(); i++) {
        if ((i > 6 && i < 11) || (i < 3) || (i > 14)) { // up and down
            neopixel.setPixelColor(i, c.getR() * b, c.getG() * b, c.getB() * b);
        }
    }
    neopixel.show();
}

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
            SetEgoNeopixelClr(pokesNeopixel[0], egoColor, fullBrightness);
            log_code(LIGHT_NORTH_CUE_EVENT);
        }
        else {
            SetAlloNeopixelClr(pokesNeopixel[0], alloColor, fullBrightness);
            log_code(LIGHT_NORTH_CUE_EVENT);
        }

    }
    else { // south
        if(is_ego_context == true){
            SetEgoNeopixelClr(pokesNeopixel[1], egoColor, fullBrightness);
            log_code(LIGHT_SOUTH_CUE_EVENT);
        }
        else {
            SetAlloNeopixelClr(pokesNeopixel[1], alloColor, fullBrightness);
            log_code(LIGHT_SOUTH_CUE_EVENT);
        }
    }
}

void go_cue_west(){
    if (is_ego_context == true) {
        SetNeopixelClr(pokesNeopixel[2], egoColor, fullBrightness);
    }
    else{
        SetNeopixelClr(pokesNeopixel[2], alloColor, fullBrightness);
    }
    log_code(LIGHT_WEST_CUE_EVENT);
}

void go_cue_east(){
    if (is_ego_context == true) {
        SetNeopixelClr(pokesNeopixel[3], egoColor, fullBrightness);
    }
    else{
        SetNeopixelClr(pokesNeopixel[3], alloColor, fullBrightness);
    }    
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

// a self terminating digital pin switch
// flipped by setting deliver_reward to true somewhere in the FSM
void punish_tone_controller(){
    
    if (trigger_punish_tone == true && punish_tone_ON == false){
        punish_tone_ON = true;
        error_cue_start = micros();
    }

    // for the supposed duration of the error cue
    else if (trigger_punish_tone == true && micros() - error_cue_start < error_cue_dur){
        spkrState = random(0,2);

        digitalWrite(SPEAKER_WEST_PIN, spkrState);
        digitalWrite(SPEAKER_EAST_PIN, spkrState);
    }

    // put it off at the end
    else if (trigger_punish_tone == true && micros() - error_cue_start > error_cue_dur){
        trigger_punish_tone = false;
        punish_tone_ON = false;
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

void update_interval_probabilities() {
    for (int i = 0; i < max_no_intervals; i++) {
        if (i < no_intervals) {
            p_short_intervals[i] = 1.0 / no_intervals;
            p_long_intervals[i]  = 1.0 / no_intervals;
        } else {
            p_short_intervals[i] = 0;
            p_long_intervals[i]  = 0;
        }
    }
}

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

void set_interval(){
    // init port -> context -> movement sampled

    // make sure they're updated
    update_interval_probabilities();

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

    // Correction loop entry
    if (!in_corr_loop) {
        if (short_north_error_counter >= corr_loop_entry && init_port == north && use_correction_loops == 1) {
            in_corr_loop = true;
            is_short_corr_loop = true;
            corr_loop_port_idx = 0;
        }
        if (short_south_error_counter >= corr_loop_entry && init_port == south && use_correction_loops == 1) {
            in_corr_loop = true;
            is_short_corr_loop = true;
            corr_loop_port_idx = 1;
        }
        if (long_north_error_counter >= corr_loop_entry && init_port == north && use_correction_loops == 1) {
            in_corr_loop = true;
            is_short_corr_loop = false;
            corr_loop_port_idx = 0;
        }
        if (long_south_error_counter >= corr_loop_entry && init_port == south && use_correction_loops == 1) {
            in_corr_loop = true;
            is_short_corr_loop = false;
            corr_loop_port_idx = 1;
        }
    }

    // If in corr loop, only sample the interval being corrected
    if (in_corr_loop) {
        init_port = (corr_loop_port_idx == 0) ? north : south;

        // short corr loop
        if (is_short_corr_loop) {
            idx = random(0, no_intervals);
            this_interval = short_intervals[idx];
            
            if (init_port == north) {
                correct_side = east;
                correct_movement = right;
            } 
            else {
                if (is_ego_context) {
                    correct_side = west;
                    correct_movement = right;
                } else {
                    correct_side = east;
                    correct_movement = left;
                }
            }
        }
        // long corr loop
        else {
            idx = random(0, no_intervals);
            this_interval = long_intervals[idx];
            if (init_port == north) {
                correct_side = west;
                correct_movement = left;
            } 
            else {
                if (is_ego_context) {
                    correct_side = east;
                    correct_movement = left;
                } else {
                    correct_side = west;
                    correct_movement = right;
                }
            }
        }
    }

    // if NOT in corr loop NOR prev broken fix, sample new mvmt and set interval based on that
    else if (in_corr_loop == false && prev_trial_broken == false){

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

    // if only broken fixation - force trial again (akin to a 1-trial corr loop)
    else if (in_corr_loop == false && prev_trial_broken == true){
        
        // short
        if (this_interval < 1500) {
            if (init_port == north) {
                correct_side = east;
                correct_movement = right;
            } 
            else {
                if (is_ego_context) {
                    correct_side = west;
                    correct_movement = right;
                } else {
                    correct_side = east;
                    correct_movement = left;
                }
            }
        }
        // long
        else {
            if (init_port == north) {
                correct_side = west;
                correct_movement = left;
            } 
            else {
                if (is_ego_context) {
                    correct_side = east;
                    correct_movement = left;
                } else {
                    correct_side = west;
                    correct_movement = right;
                }
            }
        }
    }

    // logging for analysis
    trial_counter++;
    log_int("trial_counter", trial_counter);
    log_ulong("this_interval", this_interval);
    log_int("in_corr_loop", (int) in_corr_loop);
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
                if (current_context_counter == this_context_dur){ 
                    is_ego_context = !is_ego_context; // flip it

                    // reset counter and sample new duration
                    log_int("is_ego_context", (int) is_ego_context);
                    current_context_counter = 0;

                    this_context_dur = (unsigned long) random(block_dur_min, block_dur_max);
                    log_int("this_context_dur", this_context_dur);

                    // Exit correction loop and reset counter if in correction loop
                    if (in_corr_loop) {
                        if (corr_loop_port_idx == 0 && is_short_corr_loop) short_north_error_counter = 0;
                        if (corr_loop_port_idx == 1 && is_short_corr_loop) short_south_error_counter = 0;
                        if (corr_loop_port_idx == 0 && !is_short_corr_loop) long_north_error_counter = 0;
                        if (corr_loop_port_idx == 1 && !is_short_corr_loop) long_south_error_counter = 0;
                        in_corr_loop = false;
                        log_msg("Correction loop exited due to context change");
                    }
                }
                else{ // increase counter
                    current_context_counter++;
                }

                if (in_corr_loop && use_correction_loops == 1) {
                    if (short_north_error_counter >= corr_loop_entry || long_north_error_counter >= corr_loop_entry) {
                        log_msg("In correction loop: North port");
                        init_port = north;
                    }
                    if (short_south_error_counter >= corr_loop_entry || long_south_error_counter >= corr_loop_entry) {
                        log_msg("In correction loop: South port");
                        init_port = south;
                    }

                else{
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
                    else{ 
                        // block initation port
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
                }

                trial_available_cue();
            }

            if (current_state == last_state){
                // the update loop
                if (init_port == north){
                    if (is_poking_north == true){
                        if (is_ego_context == true){
                            SetEgoNeopixelClr(pokesNeopixel[0], egoColor, dimBrightness);
                        }
                        else {
                            SetAlloNeopixelClr(pokesNeopixel[0], alloColor, dimBrightness);
                        }

                        log_code(TRIAL_ENTRY_NORTH_EVENT);
                        current_state = TRIAL_ENTRY_STATE;
                        break;
                    }
                }
                else {
                    if (is_poking_south == true){
                        if (is_ego_context == true){
                            SetEgoNeopixelClr(pokesNeopixel[1], egoColor, dimBrightness);
                        }
                        else {
                            SetAlloNeopixelClr(pokesNeopixel[1], alloColor, dimBrightness);
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

                    // animal managed to recover from jitter
                    if (jittering == true){
                        log_code(JITTER_OUT);
                        jittering = false;
                    }
                }

                if (is_poking == false) {
                    if (now()-t_poke_remain < grace_period && jittering == false){
                        jittering = true;
                        log_code(JITTER_IN);
                    }
                    
                    else if (now()-t_poke_remain > grace_period) {
                        // trial broken
                        ClearNeopixel(pokesNeopixel[0]);
                        ClearNeopixel(pokesNeopixel[1]);

                        prev_trial_broken = true; // set flag to resample trial type
                        
                        log_code(JITTER_OUT);
                        jittering = false;

                        log_code(INIT_POKEOUT_EVENT);
                        log_code(BROKEN_FIXATION_EVENT);
                        log_code(TRIAL_UNSUCCESSFUL_EVENT);
                        trigger_punish_tone = true; // trigger punish tone
                        
                        timeout_flag = 1; // set timeout delay to 1, so it goes to timeout state
                        current_state = ITI_STATE;
                        break;
                    }
                }
            }

            // if fixation is successful, go clock a choice
            if (now() - t_state_entry > this_interval){

                // in case the animal pokes out within grace period but is still a valid fixation
                if (is_poking == false){
                    log_code(INIT_POKEOUT_EVENT);

                    if (jittering == true){
                        log_code(JITTER_OUT);
                        jittering = false;
                    }
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
                init_pokeout_logged = false; // Reset flag on entry
            }

            // Detect pokeout - what if the animal does not poke out? mvmt time metric will be Nan
            if (!init_pokeout_logged && last_state == current_state) {
                if (is_poking == false) {
                    log_code(INIT_POKEOUT_EVENT);
                    init_pokeout_logged = true;
                }
            }

            // exit conditions

            // choice made
            if (is_poking_west == true || is_poking_east == true) {

                // udpate leds
                ClearNeopixel(pokesNeopixel[2]);
                ClearNeopixel(pokesNeopixel[3]);

                // correct choices
                if ((is_poking_west == true && correct_side == west) || (is_poking_east == true && correct_side == east)) {

                    reward_cue();
                    log_code(TRIAL_SUCCESSFUL_EVENT);
                    log_code(CHOICE_CORRECT_EVENT);
                    log_code(CHOICE_EVENT);
                    log_choice();

                    // Update error counters and check for correction loop exit
                    if (init_port == north) {
                        if (this_interval < timing_boundary && short_north_error_counter > 0) {
                            short_north_error_counter--;
                            if (short_north_error_counter == 0) {
                                in_corr_loop = false;
                                log_msg("Corr loop short north OFF");
                            }
                        }
                        if (this_interval >= timing_boundary && long_north_error_counter > 0) {
                            long_north_error_counter--;
                            if (long_north_error_counter == 0) {
                                in_corr_loop = false;
                                log_msg("Corr loop long north OFF");
                            }
                        }
                    } else {
                        if (this_interval < timing_boundary && short_south_error_counter > 0) {
                            short_south_error_counter--;
                            if (short_south_error_counter == 0) {
                                in_corr_loop = false;
                                log_msg("Corr loop short south OFF");
                            }
                        }
                        if (this_interval >= timing_boundary && long_south_error_counter > 0) {
                            long_south_error_counter--;
                            if (long_south_error_counter == 0) {
                                in_corr_loop = false;
                                log_msg("Corr loop long south OFF");
                            }
                        }
                    }  
                    current_state = REWARD_STATE;
                }

                // incorrect choices
                if ((correct_side == west && is_poking_east) || (correct_side == east && is_poking_west)){
                    
                    trigger_punish_tone = true; // trigger punish tone
                    log_code(TRIAL_UNSUCCESSFUL_EVENT);
                    log_code(CHOICE_INCORRECT_EVENT);
                    log_code(CHOICE_EVENT);
                    log_choice();
                    timeout_flag = 1;
                    
                    // Update error counters
                    if (init_port == north) {
                        if (this_interval < timing_boundary && short_north_error_counter < 4) short_north_error_counter++;
                        if (this_interval >= timing_boundary && long_north_error_counter < 4) long_north_error_counter++;
                    } else {
                        if (this_interval < timing_boundary && short_south_error_counter < 4) short_south_error_counter++;
                        if (this_interval >= timing_boundary && long_south_error_counter < 4) long_south_error_counter++;
                    }                 

                    current_state = ITI_STATE;
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
            if (now() - t_state_entry > this_ITI_dur){ // if ITI is over

                // if broken or incorrect go to timeout
                if (timeout_flag == 1){ 
                    current_state = TIMEOUT_STATE;
                }

                // else check if animal was recently in the port
                else if (now() - last_init_port_entry > 1000) { // one second guard
                    current_state = TRIAL_AVAILABLE_STATE;
                }
            }
            break;

        case TIMEOUT_STATE:
            // state entry
            if (current_state != last_state){
                state_entry_common();
            }

            // exit
            if (now() - t_state_entry > timeout_dur && (now() - last_init_port_entry > 1000)){ // one second guard
                timeout_flag = 0; // reset timeout flag
                current_state = TRIAL_AVAILABLE_STATE;
                break;
            }
            break;
    }
}

/*
..######..########..######...######..####..#######..##....##....########.##....##.########.
.##....##.##.......##....##.##....##..##..##.....##.###...##....##.......###...##.##.....##
.##.......##.......##.......##........##..##.....##.####..##....##.......####..##.##.....##
..######..######....######...######...##..##.....##.##.##.##....######...##.##.##.##.....##
.......##.##.............##.......##..##..##.....##.##..####....##.......##..####.##.....##
.##....##.##.......##....##.##....##..##..##.....##.##...###....##.......##...###.##.....##
..######..########..######...######..####..#######..##....##....########.##....##.########.
*/

void lights_off_controller(){

    if (lights_off == true) {
        // water
        digitalWrite(REWARD_WEST_VALVE_PIN, LOW);
        digitalWrite(REWARD_EAST_VALVE_PIN, LOW);
        digitalWrite(REWARD_PUMP_PIN, LOW);

        // speakers
        digitalWrite(SPEAKER_WEST_PIN, 1); // turn off west speaker - inverse
        digitalWrite(SPEAKER_EAST_PIN, 1); // turn off west speaker - inverse

        // BG lights
        ClearNeopixel(bgNeopixel);

        // poke lights
        for (i = 0; i < NUM_POKES; i++){
            ClearNeopixel(pokesNeopixel[i]); 
        }
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
    SetNeopixelClr(bgNeopixel, redColor, fullBrightness); // red background
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
    punish_tone_controller();
    reward_valve_controller();
    pump_controller();
    lights_off_controller(); // turn off everything when session's finished

    // sample sensors
    read_pokes();

    // serial communication with main PC
    getSerialData();
    processSerialData();
    
    // non-blocking cam sync pin
    sync_pin_controller();
}