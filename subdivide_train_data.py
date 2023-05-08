"""
Helper Function to split training data into multiple subfolders.

Although we can run the train data loader in about 10-12 minutes locally, Google Colab times out.
I believe the timeout is happening because Google Colab can't read all the files in a single directory. 
I think splitting it up so we have at most 15k files per folder should help.


Adapted from:
https://stackoverflow.com/questions/51793379/splitting-content-from-a-single-folder-to-multiple-sub-folders-using-python

"""

import os, os.path, shutil
from time import perf_counter
import datetime

#should take about 5min for train data:
folder_path = 'data/train'
new_base_path = 'data/train_subdivided'

my_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
counter = 0
sub_folder_num = 1
num_files_per_folder = 4600
mappings = []

start_time = perf_counter()

for file in my_files:
    
    if counter < num_files_per_folder:
        new_folder = 'subgroup' + str(sub_folder_num)
        counter += 1
    else:
        sub_folder_num += 1
        counter = 1
        new_folder = 'subgroup' + str(sub_folder_num)
    
    new_path = os.path.join(new_base_path, new_folder)
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    old_file_path = os.path.join(folder_path, file)
    new_file_path = os.path.join(new_path, file)
    
    shutil.copy2(old_file_path, new_file_path)
    mappings.append([file, new_folder])
    
with open("data/my_mappings.csv", "w") as f:
    for item in mappings:
        f.write(item[0] + "," + item[1] + "\n")
    
end_time = perf_counter()
print("operation took: " + str(datetime.timedelta(seconds = end_time - start_time)))


##############################################################################
