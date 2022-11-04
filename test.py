import os
import sys
import argparse
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
import dataset_processing
from model_xception import xception
from models import FakeClassifier
from models import evaluate_fake_detection as evaluate

# # # arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--model_id', help="['xception', 'meso', 'mesoinception', 'fwa', 'dsp-fwa', 'capsule']", required=True, type=int)
parser.add_argument('--resolution_id', help='[LR, HR]', required=True, type=int)
parser.add_argument('--compression_id', help="['c0', 'c23', 'c40']", required=True, type=int)
parser.add_argument('--test_resolution_id', help='[LR, HR]', required=True, type=int)
parser.add_argument('--test_compression_id', help="['c0', 'c23', 'c40']", required=True, type=int)
args=parser.parse_args()
parser.add_argument('--eval_set_id', help="['A', 'B', 'C']", default=1)
args=parser.parse_args()

#####################################################################################

set_name_list = ['A', 'B', 'C'] 
set_name_eval = set_name_list[args.eval_set_id]    # # default is Set A

resolution_list = ['LR', 'HR']
resolution = resolution_list[args.resolution_id]
test_resolution = resolution_list[args.test_resolution_id]

compression_list = ['c0', 'c23', 'c40'] # cmdline 
compression = compression_list[args.compression_id]
test_compression = compression_list[args.test_compression_id]

model_name_list = ['xception', 'meso', 'mesoinception', 'fwa', 'dsp-fwa', 'capsule'] # cmdline
model_name = model_name_list[args.model_id]

model_save_name = model_name+'_'+resolution+'_'+compression+"_new"
test_set_name = set_name_eval +'_'+test_resolution+'_'+test_compression
test_csv_name = os.path.join('csv', 'test_'+ test_set_name +'.csv')

if(set_name_eval != 'A'):
    testset = dataset_processing.FakeDetectionBCDataloader(test_csv_name)
else:
    testset = dataset_processing.FakeDetectionDataloader(test_csv_name)   ## change the csv file

testloader = DataLoader(testset,
                          batch_size=config.test_batch_size,
                          shuffle=False,
                          num_workers=8
                         )

print("Test data loaded")

num_labels = 2
model = FakeClassifier(num_labels, model_name).to(config.device)

print("Model loaded")

# for t in test_type:
# print("Test type: ", model_name + t)
models.load_model(model, model_save_name)
evaluate(model, testloader, model_save_name, True, test_set_name)
print("-------------------------------\n")
# t = test_type[2]
# model.load_model(model, model_name + t)
# print("Test type: ", model_name + t)
# evaluate(model, testloader, model_name + t, test_flag=True)

print("Done")