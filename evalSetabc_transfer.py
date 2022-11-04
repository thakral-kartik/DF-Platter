import os
import sys
import argparse
import collections
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve

################################################################################################################
# # ROC
def roc(targets, preds, save_name):
    ypred = preds[:, 1]    # # score for class 1
    fpr, tpr, _ = roc_curve(targets, ypred)
    plt.plot(fpr, tpr)
    plt.plot([0,1], [0,1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig("./plots/roc_"+save_name+".png")
    plt.clf()

# # AUC
def auc(targets, preds):
    print("ROC-AUC score: ", roc_auc_score(targets, np.argmax(preds, axis=1)))

# # Accuracy
def accuracy(targets, preds):
    print("Accuracy: ", accuracy_score(targets, np.argmax(preds, axis=1)))
    print(classification_report(targets, np.argmax(preds, axis=1), digits=4, target_names=['real', 'fake']))

################################################################################################################
def get_video_name_setA(fname):
    a = fname.split('_')
    b = '_'.join(a[:-2])
    return b

# # 1. Set A: Frame-wise Accuracy
def eval_frames_setA(fnames, targets, preds, save_name):
    accuracy(targets, preds)
    auc(targets, preds)
    roc(targets, preds, 'frame_'+save_name)    

# # dummy filename: ../frames/SetA/fsgan/HR/male/100_110/100_110_frame_0.jpg,1
# # 2. Set A: Video-wise Accuracy
def eval_videos_setA(fnames, targets, preds, save_name):
    
    # print(fnames, len(fnames))
    vnames = sorted(set([get_video_name_setA(f) for f in fnames]))
    print("Num of videos: ", len(vnames))

    new_targets = np.zeros(shape=len(vnames))
    new_preds = np.zeros(shape=(len(vnames), 2))
    counts = np.zeros(shape=len(vnames))

    for i, f in enumerate(fnames):
        vname = get_video_name_setA(f)
        ind = vnames.index(vname)
        new_targets[ind] += targets[i] 
        new_preds[ind] += preds[i] 
        counts[ind] += 1

    new_targets = new_targets/counts
    new_preds[:, 0] = new_preds[:, 0]/counts
    new_preds[:, 1] = new_preds[:, 1]/counts
    print("new_targets.shape, new_preds.shape:", new_targets.shape, new_preds.shape)

    accuracy(new_targets, new_preds)
    auc(new_targets, new_preds)
    roc(new_targets, new_preds, 'video_'+save_name)        

################################################################################################################

def get_frame_name_setBC(fname):
    a = fname.split('_')
    b = '_'.join(a[:-1])
    return b

def get_video_name_setBC(fname):
    a = fname.split('_')
    b = '_'.join(a[:-2])
    return b

# # 3. Set B, C: Face-wise. Accuracy for every face in every frame.
def eval_faces_setBC(fnames, targets, preds, save_name):
    accuracy(targets, preds)
    auc(targets, preds)
    roc(targets, preds, 'faces_'+save_name)    

# # 4. Set B, C: Frame-wise. Accuracy where correct is all faces correctly detected.
def eval_frames_setBC(fnames, targets, scores, save_name):

    preds = np.argmax(scores, axis=1)

    frame_names = sorted(set([get_frame_name_setBC(f) for f in fnames]))
    print("Num of frames: ", len(frame_names))
    # print(frame_names)

    new_targets = [[] for i in range(len(frame_names))]
    new_scores = [[] for i in range(len(frame_names))]
    new_preds = [[] for i in range(len(frame_names))]

    for i, f in enumerate(fnames):
        frame_name = get_frame_name_setBC(f)
        ind = int(frame_names.index(frame_name))
        new_targets[ind].append(targets[i])
        new_scores[ind].append(scores[i, 1]) 
        new_preds[ind].append(preds[i])

    # print(new_targets)
    # print(new_preds)

    correct = [0 for i in range(len(new_targets))]
    for i in range(len(new_targets)):
        if np.array_equal(new_targets[i], new_preds[i], equal_nan=True):
            correct[i]=1
    acc = sum(correct)/len(new_targets)
    print("Frame-wise accuracy: ", round(acc, 4))

    return frame_names, correct

# # 5. Set B, C: Video-wise where accuracy for frame is calculated based on 4.
def eval_videos_setBC(fnames, correct_preds, save_name):

    # print(fnames, len(fnames))
    vnames = sorted(set([get_video_name_setBC(f) for f in fnames]))
    print("Num of videos: ", len(vnames))

    new_preds = np.zeros(shape=(len(vnames),))
    counts = np.zeros(shape=len(vnames))

    for i, f in enumerate(fnames):
        vname = get_video_name_setA(f)
        ind = vnames.index(vname)
        new_preds[ind] += correct_preds[i]
        counts[ind] += 1
    new_preds = new_preds/counts
    new_preds = [int(p > 0.5) for p in new_preds] 
    print("Video-wise accuracy: ", round(np.average(new_preds), 4))

#########################################################################################################################################
# # MAIN
#########################################################################################################################################

# # arguments
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
model_save_name = model_name+'_'+resolution+'_'+compression
test_set_name = set_name_eval +'_'+test_resolution+'_'+test_compression

path = "predictions/"
mat_name = model_save_name+"_TEST_"+test_set_name+".mat"
d = io.loadmat(path + mat_name)
fnames = d['fnames']
targets = d['targets'].T
preds = d['preds']
# print(fnames)
print(len(fnames), targets.shape, preds.shape)

# # # # For Set A
# # eval_frames_setA(fnames, targets, preds, mat_name[:-4])
# # eval_videos_setA(fnames, targets, preds, mat_name[:-4])

# # # For Set B and Set C
eval_faces_setBC(fnames, targets, preds, mat_name[:-4])
frame_names, correct_preds = eval_frames_setBC(fnames, targets, preds, mat_name[:-4])
eval_videos_setBC(frame_names, correct_preds, mat_name[:-4])