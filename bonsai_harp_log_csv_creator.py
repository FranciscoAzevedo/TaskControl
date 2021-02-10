import os 
from pathlib import Path
import utils
import datetime
from tqdm import tqdm
import pandas as pd
import shutil
import time

# Harp log bin csv converter for all sessions given animal

animal_folder = utils.get_folder_dialog()
task_name = ['learn_to_push_cr','learn_to_push_vis_feedback']
SessionsDf = utils.get_sessions(animal_folder)

PushSessionsDf = pd.concat([SessionsDf.groupby('task').get_group(name) for name in task_name])

log_paths = [Path(path)/'bonsai_harp_log.bin' for path in PushSessionsDf['path']]

for log_path in tqdm(log_paths):

    # Only if it was not already created
    if not os.path.isfile(str(log_path.parent) + '\\bonsai_harp_log.metadata.csv'):

        # Change .txt file in the \CsvConverter.Interface folder to the session with log_path
        file = open("D:\TaskControl\Animals\\test.txt","r+")
        file.truncate(0) # erase everything

        file.write('1, ' + str(log_path))
        file.write('\n')
        file.write('200, true')

        file.close()

        # Copy file to destination folder
        shutil.copy("D:\TaskControl\Animals\\test.txt", 'D:\TaskControl\Animals\CsvConverter.Interface')

        time.sleep(120) # Sleep 2 minutes

        # Check wether status .reply file has been created before going next loop
        reply_file = open("D:\TaskControl\Animals\CsvConverter.Interface\\test.reply.txt","r")
        reply = reply_file.read()

        # In case it succeeds
        if reply.split('\n')[0] == 'ok':
            print('Successful!')
            reply_file.close()

        # In case something goes wrong
        if reply.split('\n')[0] == 'error':
            print('Error! Prob. no Harp log bin for given session or wrong path')
            reply_file.close()
