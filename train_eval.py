import argparse
import numpy as np
import os
# import tensorflow as tf
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image
import utils as ut

def save_model_by_name(model, global_step):
    save_dir = os.path.join('checkpoints', model.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, 'model-{:05d}.pt'.format(global_step))
    state = model.state_dict()
    torch.save(state, file_path)
    print('Saved to {}'.format(file_path))

def train(model, train_loader, device, tqdm, writer,
          iter_max=np.inf, iter_save=np.inf,
          model_name='model', reinitialize=False):
    # Optimization
    if reinitialize:
        model.apply(ut.reset_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    i = 0
    with tqdm(total=iter_max) as pbar:
        for i in range(iter_max):
            for batch_idx, elem in enumerate(train_loader):
                print("batch: {}".format(batch_idx))
                i += 1 # i is num of gradient steps taken by end of loop iteration
                optimizer.zero_grad()
                xu = elem['wav']
                yu = elem['label']
                xu = xu.new(xu).float().to(device)
                yu = yu.new(yu).float().to(device)
                print(xu)
                print(yu)
                loss, summaries = model.loss(xu, yu)

                loss.backward()
                optimizer.step()

                # Feel free to modify the progress bar

                pbar.set_postfix(
                    loss='{:.2e}'.format(loss))
                pbar.update(1)

                # Log summaries
                if i % 50 == 0: ut.log_summaries(writer, summaries, i)

                # Save model
                if i % iter_save == 0:
                    ut.save_model_by_name(model, i)

                if i == iter_max:
                    return

def evaluate(model, val_loader, device, batch_size, tdqm):
    with tdqm(total=len(val_loader.dataset)/batch_size) as pbar:
        for batch_idx, (xu, yu) in enumerate(val_loader):
            xu = torch.bernoulli(xu.to(device).reshape(xu.size(0), -1))
            yu = yu.new(yu).to(device).float()
            loss, summaries = model.loss(xu, yu)

            pbar.set_postfix(
                    loss='{:.2e}'.format(loss))
            pbar.update(1)

            print(loss)






