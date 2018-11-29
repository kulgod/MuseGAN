import argparse
import numpy as np
import pandas as pd
import torch
import tqdm
import os
import shututil
from pprint import pprint
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, sampler
from train import train, evaluate
import utils as ut

class WAV_Dataset(Dataset):
	def __init__(self, csv_file, wav_files, wav_map):
		self.label = pd.read_csv(csv_file)
		self.wav_files = wav_files

	def __len__(self):
		return len(self.wav_files)

	def __getitem__(self, idx):
		wav = self.wav_files[idx]
		label = self.label.iloc[idx, [1:]].as_matrix()
		sample = {'wav': wav, 'label': label}
		return sample


def delete_existing(path):
  if os.path.exists(path):
      print("Deleting existing path: {}".format(path))
      shutil.rmtree(path)


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

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	wav_dataset = WAV_Dataset(csv_file='data/annotations/labels.csv', wav_files=wav_files, wav_map=wav_map)
	dataset_size = len(wav.dataset)
	indices = list(range(dataset_size))
	split = int(np.floor(args.val_split * dataset_size))
	train_idxs, val_indxs = indices[split:], indices[:split]
	train_sampler = sampler.SubsetRandomSampler(train_indices)
	val_sampler = sampler.SubsetRandomSampler(val_indices)

	train_loader = DataLoader(wav_dataset, batch_size=args.batch_size, sampler=train_sampler)
	val_loader = DataLoader(wav_dataset, batch_size=args.batch_size, sampler=val_sampler)
	vae_lstm = VAE_LSTM(z_dim=args.z, name=model_name).to(device) #need model function

	if args.train:
	    writer = ut.prepare_writer(model_name, overwrite_existing=True)
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





