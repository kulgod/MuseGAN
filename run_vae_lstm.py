import argparse
import numpy as np
import pandas as pd
import torch
import tqdm
import os
import shutil
from pprint import pprint
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, sampler
from train_eval import train, evaluate
import utils as ut
import pickle
import MuseVAE
class WAV_Dataset(Dataset):
	def __init__(self, wav_files, labels):
		self.wav_files = wav_files
		self.labels = labels

	def __len__(self):
		return len(self.wav_files)

	def __getitem__(self, idx):
		wav = self.wav_files[idx]
		label = self.labels[idx]
		sample = {'wav': wav, 'label': label}
		return sample

if __name__ == '__main__':
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--z',         type=int, default=10,     help="Number of latent dimensions")
	parser.add_argument('--batch_size', type=int, default=20, help="Batch Size")
	parser.add_argument('--iter_max',  type=int, default=20000, help="Number of training iterations")
	parser.add_argument('--iter_save', type=int, default=10000, help="Save model every n iterations")
	parser.add_argument('--val_split', type=int, default=.1, help="Validation size")
	parser.add_argument('--run',       type=int, default=0,     help="Run ID. In case you want to run replicates")
	parser.add_argument('--train',     type=int, default=1,     help="Flag for training")
	args = parser.parse_args()
	layout = [
	    ('model={:s}',  'classifying-vae-lstm'),
	    ('z={:02d}',  args.z),
	    ('run={:04d}', args.run)
	]
	model_name = '_'.join([t.format(v) for (t, v) in layout])
	pprint(vars(args))
	print('Model name:', model_name)

	y_map = {}
	labels = pd.read_csv('data/annotations/new_labels.csv').as_matrix()
	for i in range(labels.shape[0]):
		y_map[labels[i][0]] = i

	wav_data, id_map = pickle.load(open("wav_data_1000.pkl", "rb"))
	X = np.zeros((0, wav_data.shape[1]))
	Y = np.zeros((0, 2))
	for i, song_id in enumerate(id_map):
		if song_id in y_map:
			X = np.concatenate([X, wav_data[None, i]])
			Y = np.concatenate([Y, labels[y_map[song_id]][None, 1:]])

	print(X)
	print(Y)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	wav_dataset = WAV_Dataset(wav_files=X, labels=Y)
	dataset_size = len(wav_dataset)
	print(len(wav_dataset))
	indices = list(range(dataset_size))
	split = int(np.floor(args.val_split * dataset_size))
	train_idxs, val_idxs = indices[split:], indices[:split]
	train_sampler = sampler.SubsetRandomSampler(train_idxs)
	val_sampler = sampler.SubsetRandomSampler(val_idxs)

	train_loader = DataLoader(wav_dataset, batch_size=args.batch_size, sampler=train_sampler)
	val_loader = DataLoader(wav_dataset, batch_size=args.batch_size, sampler=val_sampler)
	vae_lstm = MuseVAE.MuseVAE(x_dim=10, y_dim=2, z_dim=args.z, batch_size=args.batch_size, seq_length=100).to(device) 

	if args.train:
	    writer = ut.prepare_writer(model_name, overwrite_existing=True)
	    print("bouta train baby")
	    train(model=vae_lstm,
	          train_loader=train_loader,
	          device=device,
	          tqdm=tqdm.tqdm,
	          writer=writer,
	          iter_max=args.iter_max,
	          iter_save=args.iter_save)
	    ut.evaluate_lower_bound(vae, labeled_subset, run_iwae=args.train == 2)

	else:
	    ut.load_model_by_name(vae_lstm, global_step=args.iter_max)
	    evaluate(model=vae_lstm, val_loader=val_loader, device=device)
	    ut.evaluate_lower_bound(vae, labeled_subset, run_iwae=True)
	    # sample = vae.sample_x(200).view(200, 28, 28).unsqueeze(1)
	    # utils.save_image(sample, 'vae_sample.png')





