# # Model for training on Set A of NeurIPS deepfake dataset

import os
import sys
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader

import config
import models
from dataset_processing import FakeDetectionDataloader
from model_xception import xception
from models import FakeClassifier
from models import train_fake_detection as train

mode = 'train'

resolution_list = ['LR', 'HR']
# resolution = resolution_list[1]
resolution = resolution_list[int(sys.argv[2])]

compression_list = ['c0', 'c23', 'c40']
# compression = compression_list[2]
compression = compression_list[int(sys.argv[1])]

model_name_list = ['xception', 'meso', 'mesoinception', 'fwa', 'dsp-fwa', 'capsule']
# model_name = model_name_list[5]
model_name = str(sys.argv[3])

model_save_name = model_name+'_'+resolution+'_'+compression+"_new"
print(model_save_name)

train_csv_name = os.path.join('csv', 'train_'+resolution+'_'+compression+'_new.csv')
val_csv_name = os.path.join('csv', 'val_'+resolution+'_'+compression+'.csv')

print(train_csv_name)
# Import Dataset after preprocessing
trainset = FakeDetectionDataloader(train_csv_name)
valset = FakeDetectionDataloader(val_csv_name)
num_labels = 2

# train_size = int(0.8 * len(full_train_dataset))
# val_size = len(full_train_dataset) - train_size
# trainset, valset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])
# print("train_size, val_size: ", train_size, val_size)

trainloader = DataLoader(trainset,
                          batch_size=config.batch_size,
                          shuffle=True,
                          num_workers=16
                         )

valloader = DataLoader(valset,
                            batch_size=config.batch_size,
                            shuffle=False,
                            num_workers=16
                            )

print("data loaded")

model = FakeClassifier(num_labels, model_name)
# model = torch.nn.DataParallel(model)
model = model.to(config.device)

if config.resume_training == True:
    try:
        models.load_model(model, 'best_'+model_save_name)
        print("Resuming the training from last checkpoint")

    except:
        print("Checkpoint model not available, cannot resume training\nBegining the training from scratch ")


if mode == 'train':
    model = train(model, trainloader, valloader, model_save_name)
    models.save_model(model, model_save_name)

print("Done for ", model_save_name)


###############################################################