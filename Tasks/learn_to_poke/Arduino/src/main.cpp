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

// Parameters from Thiago Gouvea's eLife paper
unsigned long tone_dur = 150;
unsigned long tone_freq = 7500;
unsigned long reward_tone_freq = 1750;

// Parameters for current pumps and pokes
unsigned long reward_valve_dur = 2000; // more than enough for pump
unsigned long reward_pump_toggle_dur = 3; // ms
int targetToggles = 70; // Total number of toggles to perform , double of pump steps

// speaker
Tone tone_control_east;
Tone tone_control_west;

//  named variables that are easier to compare in numbers than strings
int north = 8;
int south = 2;
int west = 4;
int east = 6;

// ego centric coords
int left = 7;
int right = 9;

// non-init variables
int i;
float r; // for random processes
unsigned long this_ITI_dur;
bool spkrState;
bool pumpState;
int choice;
int correct_side;

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

    // odor
    digitalWrite(ODOR1_NORTH_VALVE_PIN, LOW); // turn off odor 1 north valve
    digitalWrite(ODOR2_NORTH_VALVE_PIN, LOW); // turn off odor 2 north valve
    digitalWrite(ODOR1_SOUTH_VALVE_PIN, LOW); // turn off odor 1 south valve
    digitalWrite(ODOR2_SOUTH_VALVE_PIN, LOW); // turn off odor 2 south valve

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

float offBrightness = 0.0;
float dimBrightness = 0.1;
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

// speaker
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

                r = random(0,1000) / 1000.0;
                if (r > 0.5){
                    correct_side = west;
                    go_cue_west();
                }
                else {
                    correct_side = east;
                    go_cue_east();
                }

                log_int("correct_side", correct_side);
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

            // sync at trial entry
            switch_sync_pin = true;
            sync_pin_controller(); // and call sync controller for enhanced temp prec.

            // choice made
            if (is_poking_west == true && correct_side == west){

                // udpate leds
                ClearNeopixel(pokesNeopixel[2]);
                ClearNeopixel(pokesNeopixel[3]);
                
                reward_cue();
                log_code(CHOICE_EVENT);
                log_choice();
                current_state = REWARD_STATE;
            }
            else if(is_poking_east == true && correct_side == east){

                // udpate leds
                ClearNeopixel(pokesNeopixel[2]);
                ClearNeopixel(pokesNeopixel[3]);
                
                reward_cue();
                log_code(CHOICE_EVENT);
                log_choice();
                current_state = REWARD_STATE;
            }

            // no choice was made
            if (now() - t_state_entry > choice_dur){
                log_code(CHOICE_MISSED_EVENT);
                log_code(TRIAL_UNSUCCESSFUL_EVENT);

                // udpate leds
                ClearNeopixel(pokesNeopixel[2]);
                ClearNeopixel(pokesNeopixel[3]);

                // cue
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
    tone_controller.begin(SPEAKER_WEST_PIN);

    pinMode(SPEAKER_EAST_PIN,OUTPUT);
    tone_controller.begin(SPEAKER_EAST_PIN);

    // ini valves 
    pinMode(REWARD_WEST_VALVE_PIN,OUTPUT);
    pinMode(REWARD_EAST_VALVE_PIN,OUTPUT);
    pinMode(REWARD_PUMP_PIN,OUTPUT);

    // ini pokes
    for (int i = 0; i < NUM_POKES; i++){
        pinMode(POKES_PINS[i],INPUT);
    }

    // ini LEDs 
    pinMode(BCKGND_LIGHTS_PIN,OUTPUT);
    bgNeopixel = Adafruit_NeoPixel(NUM_BCKGND_PIXELS,BCKGND_LIGHTS_PIN,NEO_GRB + NEO_KHZ800);
    bgNeopixel.begin();
    SetNeopixelClr(bgNeopixel, redColor, fullBrightness); // max brightness red background
    bgNeopixel.show();

    for (int i = 0; i < NUM_POKES; i++){
        pinMode(POKES_LED_PINS[i],OUTPUT);
        pokesNeopixel[i] = Adafruit_NeoPixel(NUM_LED_PIXELS, POKES_LED_PINS[i], NEO_GRB + NEO_KHZ800);
        pokesNeopixel[i].begin();
        ClearNeopixel(pokesNeopixel[i]); // clear poke
        pokesNeopixel[i].show(); // init as off
    }

    // ini odor valves 
    pinMode(ODOR1_NORTH_VALVE_PIN,OUTPUT);
    pinMode(ODOR2_NORTH_VALVE_PIN,OUTPUT);
    pinMode(ODOR1_SOUTH_VALVE_PIN,OUTPUT);
    pinMode(ODOR2_SOUTH_VALVE_PIN,OUTPUT);

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
    lights_off_controller(); // turn off everything when session's finished

    // sample sensors
    read_pokes();

    // serial communication with main PC
    getSerialData();
    processSerialData();
    
    // non-blocking cam sync pin
    sync_pin_controller();
}

