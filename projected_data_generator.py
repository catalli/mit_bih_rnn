#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import cPickle as pickle
import os
import functools
import sys

np.set_printoptions(threshold=np.nan)

script_path = os.path.dirname(os.path.realpath(__file__))

data_path = ''.join([script_path, "/stdb_delight.pkl"])

reuse_dict = True

if len(sys.argv) > 1:
	max_b_length = int(sys.argv[1])
        data_save_path = ''.join([script_path, "/stdb_projected_", sys.argv[1], ".pkl"])
        reuse_dict = False
else:
	data_save_path = ''.join([script_path, "/stdb_projected.pkl"])

_dict_path = ''.join([script_path, '/stdb_delight_dict.pkl'])

if reuse_dict:
    dict_load = open(_dict_path, 'r')
    delight_b = pickle.load(dict_load)
    dict_load.close()
    max_b_length = len(delight_b)


data_file = open(data_path, 'r')

data = pickle.load(data_file)

data_file.close()

window_length = 100

no_features = window_length * len(data[0][0][0])

window_skip = no_features/max_b_length

no_epochs = 400
no_centroids = 4
no_macro_epochs = 10

#Non-patient-specific-training error target from state-of-the-art of 2017: http://ieeexplore.ieee.org/document/7893269/
error_target = 2.0
clustered_error_target = 5.0

def feed_windows(_data, _window_skip, _window_len, _features_per_step):
    data_seq = np.zeros((len(_data)/_window_skip,_window_len*_features_per_step))
    window_start_index = 0
    window_end_index = window_start_index+_window_len
    in_seq_index = 0
    while window_end_index < len(_data):
        data_window = _data[window_start_index:window_end_index].flatten()
        data_seq[in_seq_index] = data_window
        in_seq_index+=1
        window_start_index+=_window_skip
        window_end_index+=_window_skip
    return data_seq

data_train_or_test = data[2]

no_train = len(data_train_or_test)-np.sum(data_train_or_test)
no_test = np.sum(data_train_or_test)

true_data = [[np.zeros((no_train, len(data[0][0])/window_skip, no_features),dtype=np.float32), np.zeros((no_train, len(data[1][0])),dtype=np.float32)],[np.zeros((no_test, len(data[0][0])/window_skip, no_features),dtype=np.float32), np.zeros((no_test, len(data[1][0])),dtype=np.float32)]]

print("no_train: ",no_train,"\nno_test: ",no_test)

test_index = 0
train_index = 0

for i in range(len(data_train_or_test)):
	if data_train_or_test[i]:
		true_data[1][0][test_index] = feed_windows(data[0][i],window_skip,window_length,len(data[0][0][0]))
		true_data[1][1][test_index] = data[1][i]
		test_index+=1
	else:
		true_data[0][0][train_index] = feed_windows(data[0][i],window_skip,window_length,len(data[0][0][0]))
                true_data[0][1][train_index] = data[1][i]
                train_index+=1

for i in range(len(true_data[0][0])):
    all_zero_index = 0
    for j in range(len(true_data[0][0][0])):
        if max(true_data[0][0][i][j]) == 0.0 and min(true_data[0][0][i][j]) == 0.0:
            all_zero_index = j
            break
    if all_zero_index > no_features/len(data[0][0][0])-1:
        true_data[0][0][i][all_zero_index-no_features/len(data[0][0][0])/window_skip:] = np.zeros((len(data[0][0])/window_skip-(all_zero_index-no_features/len(data[0][0][0])/window_skip), no_features),dtype=np.float32)

for i in range(len(true_data[1][0])):
    all_zero_index = 0
    for j in range(len(true_data[1][0][0])):
        if max(true_data[1][0][i][j]) == 0.0 and min(true_data[1][0][i][j]) == 0.0:
            all_zero_index = j
            break
    if all_zero_index > no_features/len(data[0][0][0])-1:
        true_data[1][0][i][all_zero_index-no_features/len(data[0][0][0])/window_skip:] = np.zeros((len(data[0][0])/window_skip-(all_zero_index-no_features/len(data[0][0][0])/window_skip), no_features),dtype=np.float32)

if not reuse_dict:
    delight_threshold = 8.0e-7

    delight_data_train_x = true_data[0][0].reshape((true_data[0][0].shape[0]*true_data[0][0].shape[1],true_data[0][0].shape[2]))

    delight_b = np.zeros((max_b_length,no_features),dtype=np.float32)

    delight_b_vacancy = True

    b_alternatives = np.zeros((max_b_length, max_b_length, no_features), dtype=np.float32)

    b_alt_errors = np.zeros((max_b_length+1), dtype=np.float32)

    def delight_error(anew, b):
        projection= np.matmul(b.T,anew)
        inv=np.linalg.pinv(np.dot(b.T, b))
        inv = np.matmul(b, inv)
        reconstruction = np.matmul(inv, projection)
        error = reconstruction-anew
        #print("recon: ", reconstruction, "\nanew: ", anew)
        return np.square(error).sum()/np.square(anew).sum()



    for i in range(len(delight_data_train_x)):
        if max(delight_data_train_x[i]) > 0.0 or min(delight_data_train_x[i]) < 0.0:
            print("Evaluating window no. ", i ," of ", len(delight_data_train_x))
            #print("delight_b: ", delight_b)
            if not delight_b_vacancy:
                for j in range(max_b_length):
                    b_alternatives[j] = delight_b
                    b_alternatives[j][j] = delight_data_train_x[i]/np.linalg.norm(delight_data_train_x[i])
            for j in range(max_b_length):
                if delight_b_vacancy:
                    if j == max_b_length-1:
                        delight_b_vacancy = False
                    if max(delight_b[j]) == 0.0:
                        delight_b[j] = delight_data_train_x[i]/np.linalg.norm(delight_data_train_x[i])
                        break
                else:
                    b_alt_errors[j] = delight_error(delight_data_train_x[i].reshape((1,delight_data_train_x[i].shape[0])).transpose(),b_alternatives[j].T)
            if not delight_b_vacancy:
                    b_alt_errors[max_b_length] = delight_error(delight_data_train_x[i].reshape((1,delight_data_train_x[i].shape[0])).transpose(),delight_b.T)
                    print("b_alt_errors: ", b_alt_errors)
                    if np.argmin(b_alt_errors) != max_b_length and np.sqrt(b_alt_errors[max_b_length]) > delight_threshold:
                        print("Replacing row ", np.argmin(b_alt_errors), " with window ", i)
                        delight_b = b_alternatives[np.argmin(b_alt_errors)]

    dict_save = open(_dict_path, "w")
    pickle.dump(delight_b, dict_save)
    dict_save.close()


true_data[0][0] = np.matmul(true_data[0][0], delight_b.T)
true_data[1][0] = np.matmul(true_data[1][0], delight_b.T)

savefile = open(data_save_path, 'w')
pickle.dump(true_data, savefile)
savefile.close()
