import torch
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# num_labels = 2
batch_size = 128
test_batch_size = 64

lr = 0.0001
num_epochs = 25

# resume_training = True
resume_training = False

real = 0
fake = 1