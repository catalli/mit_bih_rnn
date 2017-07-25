#!/usr/bin/env python
from __future__ import print_function
import os
import wfdb
import numpy as np
import pickle

db_dir = "/mit-bih-arr/"
script_path = os.path.dirname(os.path.realpath(__file__))
dl_path = ''.join([script_path, db_dir])
output_path = ''.join([script_path, '/mit_bih.pkl'])
print("Reading data from ",dl_path)
#wfdb.dldatabase("mitdb", dl_path)
file_names = []
for root, dirs, filenames in os.walk(dl_path):
	for f in filenames:
		if '.dat' in f:
			file_path = ''.join([dl_path,f.split(".")[0]])
			file_names.append(file_path)


sequence_lengths = []
no_seqs = 0

sampling_divider = 20

beat_classes = ['N', 'L', 'R', 'A', 'a', 'J', 'S', 'V', 'F', 'e', 'j', 'E', '/', 'f','!']

#Classes collapsed and quotas set based on https://www.researchgate.net/figure/51772483_tbl1_Table-1-Mapping-of-MIT-BIH-arrhythmia-database-heartbeat-types-to-the-AAMI-heartbeat
aami_classes = ['N', 'S', 'V', 'F']

class_map = {'N':'N','L':'N','R':'N','A':'S','a':'S','J':'S','S':'S','V':'V','F':'F','e':'N','j':'N','E':'V','/':'N','f':'N','!':'V'}

relevant_anns = []

all_sequence_anns = []

seq_quota = 30000

type_quotas = [600,600,600,600,148,84,2,600,600,16,228,106,600,600,472]

test_quotas = [250,250,250,250,74,41,1,250,250,8,113,53,250,250,236]
for f in file_names:
	record = wfdb.rdsamp(f)
	annotation = wfdb.rdann(f, 'atr')
	beat_anns = [record.recordname]
	real_beat_anns = [record.recordname]
	for a in range(len(annotation.annsamp)):
		if annotation.anntype[a] in beat_classes:
			beat_anns.append([annotation.annsamp[a], annotation.anntype[a]])
	for a in range(1, len(beat_anns)):
		choice = 0 #np.random.randint(0,10)
		if choice < 9 and no_seqs <= seq_quota and type_quotas[beat_classes.index(beat_anns[a][1])] > 0:
			if a == 1:
				sequence_lengths.append(beat_anns[a][0]+1)
				beat_anns[a].append(beat_anns[a][0]+1)
			else:
				sequence_lengths.append(beat_anns[a][0]-beat_anns[a-1][0]+1)
				beat_anns[a].append(beat_anns[a][0]-beat_anns[a-1][0]+1)
			no_seqs+=1
			type_quotas[beat_classes.index(beat_anns[a][1])]-=1
			real_beat_anns.append(beat_anns[a])
	all_sequence_anns.append(real_beat_anns)

max_len = max(sequence_lengths)

print("no_seqs: ",no_seqs," max_len: ",max_len)

data = [np.zeros((no_seqs,max_len/sampling_divider+1,2), dtype=np.float32), np.zeros((no_seqs, len(aami_classes)), dtype=np.float32), np.zeros((no_seqs),dtype=np.int32)]

bih_anns_in_order = []

data_index = 0

for f in file_names:
	print(data_index)
	if data_index >= len(data[0]):
		break
	record = wfdb.rdsamp(f)
	annotation, wfdb.rdann(f, 'atr')
	this_seq_anns = []
        ann_dict = {}
	for a in all_sequence_anns:
		if record.recordname in a:
			for b in a:
				if b != record.recordname:
					this_seq_anns.append(b)
			break
	for ann in this_seq_anns:
		if data_index < len(data[0]):
			for r in range(ann[0]-ann[2]+1, ann[0]+1, sampling_divider):
				print((r-(ann[0]-ann[2]+1))/sampling_divider)
				data[0][data_index][(r-(ann[0]-ann[2]+1))/sampling_divider][0] = record.p_signals[r][0]
				data[0][data_index][(r-(ann[0]-ann[2]+1))/sampling_divider][1] = record.p_signals[r][1]
				beat_ann = class_map[ann[1]]
			if beat_ann != 'Q' and beat_ann != '':
				data[1][data_index][aami_classes.index(beat_ann)] = 1.0
			bih_anns_in_order.append(ann[1])
			data_index+=1

for i in range(len(data[2])):
        choice = np.random.randint(0,2)
        if choice == 0 and max(data[1][i]) > 0.9 and test_quotas[beat_classes.index(bih_anns_in_order[i])] > 0:
                data[2][i] = 1
                test_quotas[beat_classes.index(bih_anns_in_order[i])]-=1

output_file = open(output_path, 'w')

pickle.dump(data, output_file)

output_file.close()
