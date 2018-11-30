import pickle
# def convert(num):
# 	return (num-1) * .25 - 1

# import csv
# with open("new_labels.csv", 'w') as mf:
# 	writer = csv.writer(mf)
# 	writer.writerow(['song_id', 'mean_arousal', 'mean_valence'])
# 	with open ("static_annotations.csv", 'r') as f: 
# 		reader = csv.reader(f)
# 		for line in reader:
# 			if line[0] == 'song_id':
# 				continue
# 			new_line = []
# 			old_val, old_arous = float(line[1]), float(line[3])
# 			new_arous = convert(old_arous)
# 			new_val = convert(old_val)
# 			writer.writerow([line[0], new_arous, new_val])




with open("wav_data_1000.pkl", 'rb') as f:
	x = pickle.load(f)
	print(x)