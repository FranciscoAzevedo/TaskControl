// basic outline of the reader taken from
// http://forum.arduino.cc/index.php?topic=396450.0

#include <Arduino.h>
#include "interface_variables.h"

// this line limits total command length to 128 chars - adjust if necessary (very long var names)
const byte numChars = 128;
char receivedChars[numChars];
char buf[numChars];

bool newData = false;
bool verbose = true;
bool run = false;
bool deliver_reward_west = false;
bool deliver_reward_east = false;
bool togglingActive = false;
unsigned long previousMillis = 0; // Tracks the last time the pin was toggled
bool punish = false;

int current_state = 0; // WATCH OUT this is ini state

// fwd declare functions for logging
unsigned long now();
void log_bool(const char name[], bool value);
void log_int(const char name[], int value);
// void log_long(const char name[], long value);
void log_ulong(const char name[], unsigned long value);
void log_float(const char name[], float value);

void getSerialData() {
    // check if command data is available and if yes read it
    // all commands are flanked by <>

    static boolean recvInProgress = false;
    static byte ndx = 0;
    char startMarker = '<';
    char endMarker = '>';
    char rc;
 
    // loop that reads the entire command
    while (Serial.available() > 0 && newData == false) {
        rc = Serial.read();

        // read until end marker
        if (recvInProgress == true) {
            if (rc != endMarker) {
                receivedChars[ndx] = rc;
                ndx++;
                if (ndx >= numChars) {
                    ndx = numChars - 1;
                }
            }
            else {
                receivedChars[ndx] = '\0'; // terminate the string
                recvInProgress = false;
                ndx = 0;
                newData = true;
            }
        }

        // enter reading if startmarker received
        else if (rc == startMarker) {
            recvInProgress = true;
        }
    }
}

void processSerialData() {
    if (newData == true) {
        // echo back command if verbose
        if (verbose==true) {
            snprintf(buf, sizeof(buf), "<Arduino received: %s>", receivedChars);
            Serial.println(buf);
        }

        // get total length of message
        unsigned int len = 0;
        for (unsigned int i = 0; i < numChars; i++){
            if (receivedChars[i] == '\0'){
                len = i;
                break;
            }
        }

        // GET SET CMD 
        char mode[4];
        strlcpy(mode, receivedChars, 4);

        // GET
        if (strcmp(mode,"GET")==0){
            char varname[len-4+1];
            strlcpy(varname, receivedChars+4, len-4+1);

        // INSERT_GETTERS

        if (strcmp(varname,"ITOI_dur")==0){
            log_ulong("ITOI_dur", ITOI_dur);
        }

        if (strcmp(varname,"timeout_dur")==0){
            log_ulong("timeout_dur", timeout_dur);
        }

        if (strcmp(varname,"choice_dur")==0){
            log_ulong("choice_dur", choice_dur);
        }

        if (strcmp(varname,"t_init_max")==0){
            log_ulong("t_init_max", t_init_max);
        }

        if (strcmp(varname,"mean_fix_dur")==0){
            log_ulong("mean_fix_dur", mean_fix_dur);
        }

        if (strcmp(varname,"inc_fix_dur")==0){
            log_ulong("inc_fix_dur", inc_fix_dur);
        }

        if (strcmp(varname,"dec_fix_dur")==0){
            log_ulong("dec_fix_dur", dec_fix_dur);
        }

        if (strcmp(varname,"init_port_blocks")==0){
            log_ulong("init_port_blocks", init_port_blocks);
        }

        if (strcmp(varname,"port_dur_min")==0){
            log_ulong("port_dur_min", port_dur_min);
        }

        if (strcmp(varname,"port_dur_max")==0){
            log_ulong("port_dur_max", port_dur_max);
        }

        }

        // SET
        if (strcmp(mode,"SET")==0){
            char line[len-4+1];
            strlcpy(line, receivedChars+4, len-4+1);

            // get index of space
            len = sizeof(line)/sizeof(char);
            unsigned int split = 0;
            for (unsigned int i = 0; i < numChars; i++){
                if (line[i] == ' '){
                    split = i;
                    break;
                }
            }

            // split by space
            char varname[split+1];
            strlcpy(varname, line, split+1);

            char varvalue[len-split+1];
            strlcpy(varvalue, line+split+1, len-split+1);

            // INSERT_SETTERS

        if (strcmp(varname,"ITOI_dur")==0){
            ITOI_dur = strtoul(varvalue,NULL,10);
        }

        if (strcmp(varname,"timeout_dur")==0){
            timeout_dur = strtoul(varvalue,NULL,10);
        }

        if (strcmp(varname,"choice_dur")==0){
            choice_dur = strtoul(varvalue,NULL,10);
        }

        if (strcmp(varname,"t_init_max")==0){
            t_init_max = strtoul(varvalue,NULL,10);
        }

        if (strcmp(varname,"mean_fix_dur")==0){
            mean_fix_dur = strtoul(varvalue,NULL,10);
        }

        if (strcmp(varname,"inc_fix_dur")==0){
            inc_fix_dur = strtoul(varvalue,NULL,10);
        }

        if (strcmp(varname,"dec_fix_dur")==0){
            dec_fix_dur = strtoul(varvalue,NULL,10);
        }

        if (strcmp(varname,"init_port_blocks")==0){
            init_port_blocks = strtoul(varvalue,NULL,10);
        }

        if (strcmp(varname,"port_dur_min")==0){
            port_dur_min = strtoul(varvalue,NULL,10);
        }

        if (strcmp(varname,"port_dur_max")==0){
            port_dur_max = strtoul(varvalue,NULL,10);
        }

        }

        // UPD - update trial probs - HARDCODED for now, n trials
        // format UPD 0 0.031 or similar
        // if (strcmp(mode,"UPD")==0){
            
        //     char line[len-4+1];
        //     strlcpy(line, receivedChars+4, len-4+1);

        //     // get index of space
        //     len = sizeof(line)/sizeof(char);
        //     unsigned int split = 0;
        //     for (unsigned int i = 0; i < numChars; i++){
        //         if (line[i] == ' '){
        //             split = i;
        //             break;
        //         }
        //     }

        //     // split by space
        //     char varname[split+1];
        //     strlcpy(varname, line, split+1);

        //     char varvalue[len-split+1];
        //     strlcpy(varvalue, line+split+1, len-split+1);

        //     int ix = atoi(varname);
        //     float p = atof(varvalue);
        //     p_interval[ix] = p;
        // }

        // CMD
        if (strcmp(mode,"CMD")==0){
            char CMD[len-4+1];
            strlcpy(CMD, receivedChars+4, len-4+1);

            // manually implement functions here

            // Stop and Go functionality
            if (strcmp(CMD,"RUN")==0){
                run = true;
                Serial.println("<Arduino is running>");
            }

            if (strcmp(CMD,"HALT")==0){
                run = false;
                Serial.println("<Arduino is halted>");
            }

            if (strcmp(CMD,"w")==0){
                deliver_reward_west = true;
                togglingActive = true;
                previousMillis = millis();
            }

            if (strcmp(CMD,"e")==0){
                deliver_reward_east = true;
                togglingActive = true;
                previousMillis = millis();
            }

            if (strcmp(CMD,"p")==0){
                punish = true;
            }
        }

        newData = false;
    }
}
