"""
Get some basic stats on the datasets.

Assumes we have the train, test, and val DataLoaders
already loaded to the local environment
"""
import os
import numpy as np
from collections import Counter

data_path = './data/'  
my_subject_ids = []
my_length_of_stays = []
test_labels = []
val_labels = []
train_labels = []

for i in range(len(test_data_loader._data["X"])):
    my_subject_ids.append(test_data_loader._data["name"][i].split('_')[0])
    my_length_of_stays.append(test_data_loader._data["ts"][i][-1])
    test_labels.append(test_data_loader._data["ys"][i])
for i in range(len(val_data_loader._data["X"])):
    my_subject_ids.append(val_data_loader._data["name"][i].split('_')[0])
    my_length_of_stays.append(val_data_loader._data["ts"][i][-1])
    val_labels.append(val_data_loader._data["ys"][i])
for i in range(len(train_data_loader._data["X"])):
    my_subject_ids.append(train_data_loader._data["name"][i].split('_')[0])
    my_length_of_stays.append(train_data_loader._data["ts"][i][-1])
    train_labels.append(train_data_loader._data["ys"][i])
 

#Breakdown for # patients and stays:
print("Total number of ICU stays: " + str(len(my_subject_ids)))
print("Total number of patients: " + str(len(list(set(my_subject_ids)))))
print("--- max number of stays per patient: " + str(max(Counter(my_subject_ids).values())))
print("--- max length of stay: " + str(max(my_length_of_stays)))
#print("--- max length of stay in days: " + str(max(my_length_of_stays)/24))
print("--- min number of stays per patient: " + str(min(Counter(my_subject_ids).values())))
print("--- min length of stay: " + str(min(my_length_of_stays)))
print("--- average number of stays per patient: " + str(sum(Counter(my_subject_ids).values()) / len(Counter(my_subject_ids))))
print("--- average length of stay: " + str(sum(my_length_of_stays) / len(my_length_of_stays)))

                                               
#Breakdowns of train/test/val
total_test_visits = 0
total_pos_test_visits = 0
total_val_visits = 0
total_pos_val_visits = 0
total_train_visits = 0
total_pos_train_visits = 0

for i in range(len(test_labels)):
    test_labels[i] = [int(x) for x in test_labels[i]]
    total_test_visits += len(test_labels[i])
    total_pos_test_visits += sum(test_labels[i])
for i in range(len(val_labels)):
    val_labels[i] = [int(x) for x in val_labels[i]]
    total_val_visits += len(val_labels[i])
    total_pos_val_visits += sum(val_labels[i])
for i in range(len(train_labels)):
    train_labels[i] = [int(x) for x in train_labels[i]]
    total_train_visits += len(train_labels[i])
    total_pos_train_visits += sum(train_labels[i])

total_visits = total_test_visits + total_val_visits + total_train_visits
total_pos_visits = total_pos_test_visits + total_pos_val_visits + total_pos_train_visits

#Total number of visits for *loaded* subsets:
print("Total number of visits (loaded subset): " + str(total_visits) + ", Positive samples: " + str(total_pos_visits))
 
print("--Train--")
print("Number of stays: " + str(len(train_data_loader._data["X"])) + ", Number of visits: " + str(total_train_visits) + ", Number positive visits: " + str(total_pos_train_visits))
print("--Validation--")
print("Number of stays: " + str(len(val_data_loader._data["X"])) + ", Number of visits: " + str(total_val_visits) + ", Number positive visits: " + str(total_pos_val_visits))
print("--Test--")
print("Number of stays: " + str(len(test_data_loader._data["X"])) + ", Number of visits: " + str(total_test_visits) + ", Number positive visits: " + str(total_pos_test_visits))


#Breakdown for total number of visits + labels for FULL datatset, not what was loaded and used:
with open(os.path.join(data_path, 'train_listfile.csv'), "r") as lfile:
    my_data1 = lfile.readlines()[1:]  # skip the header
with open(os.path.join(data_path, 'val_listfile.csv'), "r") as lfile:
    my_data2 = lfile.readlines()[1:]  # skip the header
with open(os.path.join(data_path, 'test_listfile.csv'), "r") as lfile:
    my_data3 = lfile.readlines()[1:]  # skip the header

len(my_data1) + len(my_data2) + len(my_data3)   #total number of "visits" where a visit is each instance where
                                                # data is collected within a given ICU stay

print("Total number of visits (full dataset): " + str(len(my_data1) + len(my_data2) + len(my_data3)))


###############################################################################
###If we want more specific demographic data like age, weight, gender,etc, we could look into 
###C:\Users\joann\OneDrive\Documents\UIUC\CS598_DL4Healthcare\Project\MedFuse\mimic4extract\data\root\train\*\episodeX
###############################################################################
