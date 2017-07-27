#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import cPickle as pickle
import os
import functools
import sys

np.set_printoptions(threshold=np.nan)

script_path = os.path.dirname(os.path.realpath(__file__))

data_path = ''.join([script_path, "/chfdb_delight.pkl"])

reuse_dict = True

if len(sys.argv) > 1:
	sampling_divider = int(sys.argv[1])
        data_save_path = ''.join([script_path, "/chfdb_subsampled_", sys.argv[1], ".pkl"])
        reuse_dict = False
else:
	data_save_path = ''.join([script_path, "/chfdb_subsampled.pkl"])
        sampling_divider = 20


data_file = open(data_path, 'r')

data = pickle.load(data_file)

data_file.close()

window_length = 100

window_skip = sampling_divider

no_features = window_length * len(data[0][0][0])

no_epochs = 400
no_centroids = 4
no_macro_epochs = 10

#Non-patient-specific-training error target from state-of-the-art of 2017: http://ieeexplore.ieee.org/document/7893269/
error_target = 2.0
clustered_error_target = 5.0

def feed_windows(_data, _window_skip, _window_len, _features_per_step, _sampling_divider):
    data_seq = np.zeros((len(_data)/_window_skip,(_window_len/_sampling_divider)*_features_per_step))
    window_start_index = 0
    window_end_index = window_start_index+_window_len
    in_seq_index = 0
    while window_end_index < len(_data):
        data_window = _data[window_start_index:window_end_index:_sampling_divider].flatten()
        data_seq[in_seq_index] = data_window
        in_seq_index+=1
        window_start_index+=_window_skip
        window_end_index+=_window_skip
    return data_seq

data_train_or_test = data[2]

no_train = len(data_train_or_test)-np.sum(data_train_or_test)
no_test = np.sum(data_train_or_test)

true_data = [[np.zeros((no_train, len(data[0][0])/window_skip, no_features/sampling_divider),dtype=np.float32), np.zeros((no_train, len(data[1][0])),dtype=np.float32)],[np.zeros((no_test, len(data[0][0])/window_skip, no_features/sampling_divider),dtype=np.float32), np.zeros((no_test, len(data[1][0])),dtype=np.float32)]]

print("no_train: ",no_train,"\nno_test: ",no_test)

test_index = 0
train_index = 0

for i in range(len(data_train_or_test)):
	if data_train_or_test[i]:
		true_data[1][0][test_index] = feed_windows(data[0][i],window_skip,window_length,len(data[0][0][0]), sampling_divider)
		true_data[1][1][test_index] = data[1][i]
		test_index+=1
	else:
		true_data[0][0][train_index] = feed_windows(data[0][i],window_skip,window_length,len(data[0][0][0]), sampling_divider)
                true_data[0][1][train_index] = data[1][i]
                train_index+=1

for i in range(len(true_data[0][0])):
    all_zero_index = 0
    for j in range(len(true_data[0][0][0])):
        if max(true_data[0][0][i][j]) == 0.0 and min(true_data[0][0][i][j]) == 0.0:
            all_zero_index = j
            break
    if all_zero_index > no_features/len(data[0][0][0])-1:
        true_data[0][0][i][all_zero_index-no_features/len(data[0][0][0])/window_skip:] = np.zeros((len(data[0][0])/window_skip-(all_zero_index-no_features/len(data[0][0][0])/window_skip), no_features/sampling_divider),dtype=np.float32)

for i in range(len(true_data[1][0])):
    all_zero_index = 0
    for j in range(len(true_data[1][0][0])):
        if max(true_data[1][0][i][j]) == 0.0 and min(true_data[1][0][i][j]) == 0.0:
            all_zero_index = j
            break
    if all_zero_index > no_features/len(data[0][0][0])-1:
        true_data[1][0][i][all_zero_index-no_features/len(data[0][0][0])/window_skip:] = np.zeros((len(data[0][0])/window_skip-(all_zero_index-no_features/len(data[0][0][0])/window_skip), no_features/sampling_divider),dtype=np.float32)

savefile = open(data_save_path, 'w')
pickle.dump(true_data, savefile)
savefile.close()
