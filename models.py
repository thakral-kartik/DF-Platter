import os
import numpy as np
from tqdm import tqdm
import scipy.io as io
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve

import torch
import torch.nn as nn

import config
import model_xception
from model_fwa import ResNet, SPPNet
from model_meso import Meso4, MesoInception4
from model_capsule_bkp import VggExtractor, CapsuleNet

class FakeClassifier(nn.Module):
    def __init__(self, num_labels=2, network_name='xception'):
        super(FakeClassifier, self).__init__()
        self.num_labels = num_labels
        
        if network_name == 'xception':
            self.model = model_xception.xception(num_classes=1000, pretrained=False)
            self.model_wo_fc = nn.Sequential(*(list(self.model.children())[:-1]))
            self.flatten_size = 100352
        elif network_name == 'meso':
            self.model_wo_fc = Meso4()
            self.flatten_size = 16*7*7
        elif network_name == 'mesoinception':
            self.model_wo_fc = MesoInception4()
            self.flatten_size = 16*7*7
        elif network_name == 'fwa':
            self.model_wo_fc = ResNet(layers=34, pretrained=False)
            self.flatten_size = 512*7*7
        elif network_name == 'dsp-fwa':
            self.model_wo_fc = SPPNet()
            self.flatten_size = 20992
        elif network_name == 'capsule':
            self.model_wo_fc = nn.Sequential(VggExtractor(train=False), CapsuleNet(4, 0))
            self.flatten_size = 16
        else:
            print("Model not supported..")
            quit()
        
        
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.flatten_size, out_features=512),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512, out_features=128),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=128, out_features=self.num_labels)
        )

    def forward(self, x):
        x = self.model_wo_fc(x)
        # print("features.shape:", x.shape)
        x = torch.flatten(x, 1)
        # print("flatten_features.shape:", x.shape)
        x = self.fc(x)
        # print("Predictions.shape:", x.shape)

        return x

def init_loss():
    # return torch.nn.BCELoss()
    return torch.nn.CrossEntropyLoss()

def save_model(model, name):
    if not os.path.isdir('saved_models'):
        os.mkdir('saved_models')
    
    torch.save(model.state_dict(), os.path.join('saved_models', name+'.pt'))
    print("\nModel successfully saved.")
            
def load_model(model, name):
    new_d = {}
    d = torch.load(os.path.join('saved_models', name+'.pt'))
    for k in d.keys():
        newk = k.replace("module.", "")
        new_d[newk] = d[k]
    model.load_state_dict(new_d)
    return model

###############################################################################
######                           Fake Detection                          ######
###############################################################################


def train_fake_detection(model, trainloader, valloader, model_save_name):
    loss_function = init_loss().to(config.device)
    # model = torch.nn.DataParallel(model, device_ids=[0,3])
    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])

    model = model.to(config.device)
    model.train()
    
    # optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9,0.999),eps=1e-08,weight_decay=0,amsgrad=False)
    
    # # parameters for early stopping
    early_stop_max = 3  # # hyperparameter
    early_accuracy = 0.0
    early_count = 0
    early_model = None
    best_acc = 0

    softmax = torch.nn.Softmax(dim=1)

    for epoch in range(1, config.num_epochs+1):
        d = {'preds':[], 'targets':[]}
        running_loss, running_accuracy = 0, 0
        print("Epoch {}/{}".format(epoch, config.num_epochs))
        
        with tqdm(total=len(trainloader)) as pbar:
            for batch_idx, batch in enumerate(trainloader):
                fnames, images, labels = batch
                images = images.to(config.device)
                targets = labels.long().to(config.device)
                
                outputs = model(images)
                #print("output.shape:", output.shape, "targets.shape:", targets.shape)
                #print("output.shape:", output.shape, "target.shape", targets.shape, "output[0]:", output[0])
                
                loss = loss_function(outputs, targets)
                #print("images.shape:", images.shape, "targets.shape:", targets.shape, "output.shape:", output.shape)
                
                optimizer.zero_grad()
                loss.backward()
                # loss.mean().backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                preds = softmax(outputs)
                batch_correct = torch.sum(torch.argmax(preds, dim=1) == targets)
                batch_accuracy = batch_correct/images.size(0)
                running_accuracy += batch_accuracy

                d['preds'].extend(preds.detach().cpu().numpy())
                d['targets'].extend(targets.detach().cpu().numpy())

                pbar.set_postfix(Epochs='{}'.format(epoch),
                                    Loss='{0:10f}'.format(loss.item()),
                                    Acc='{0:.4f}'.format(batch_accuracy))
                
                pbar.update(1)
            
        t = np.array(d['targets'])
        p = np.array(d['preds'])

        epoch_loss = running_loss/len(trainloader)
        print("\nEpoch loss:", epoch_loss)
        print("\nEpoch Accuracy:", running_accuracy.item()/(len(trainloader)*config.batch_size), accuracy_score(d['targets'], np.argmax(d['preds'], axis=1)))
        # print("\nClassification report:\n", classification_report(d['targets'], d['preds'], digits=4))
        
        print("\nValidation Set: ")
        acc = evaluate_fake_detection(model, valloader, model_save_name)
        if early_accuracy < acc:
            early_accuracy = acc
            early_model = model
        else:
            early_count += 1

        if acc >= best_acc:
            best_acc = acc
            save_model(model, 'best_'+model_save_name)

        print("#"*50, "\n")
        
        if early_count >= early_stop_max:
            print("Validation accuracy did not improve for ", early_stop_max, " epochs.")
            break

            # d['loss'].append(epoch_loss)
    return early_model

def evaluate_fake_detection(model, testloader, model_save_name=None, test_flag=False, test_set_name=None):
    model = model.to(config.device)
    model.eval()
    
    running_loss, running_accuracy = 0, 0
    d = {'fnames':[], 'preds':[], 'targets':[]}

    softmax = torch.nn.Softmax(dim=1)

    with tqdm(total=len(testloader)) as pbar:
        for batch_idx, batch in enumerate(testloader):
            fnames, images, labels = batch
            images = images.to(config.device)
            targets = labels.float().to(config.device)
            
            outputs = model(images)
            preds = softmax(outputs)
            batch_correct = torch.sum(torch.argmax(preds, dim=1) == targets)
            batch_accuracy = batch_correct/images.size(0)
            running_accuracy += batch_accuracy
            
            d['fnames'].extend(list(fnames))
            d['preds'].extend(preds.detach().cpu().numpy())
            d['targets'].extend(targets.detach().cpu().numpy())

            pbar.set_postfix(Acc='{0:.4f}'.format(batch_accuracy))
            pbar.update(1)
    
    # print(t.shape, p.shape)
    print("\nEpoch Accuracy:", running_accuracy.item()/(len(testloader)*config.batch_size), accuracy_score(d['targets'], np.argmax(d['preds'], axis=1)))
    print("\nClassification report:\n", classification_report(d['targets'], np.argmax(d['preds'], axis=1), digits=4))
    # print("\nROC-AUC score: ", roc_auc_score(d['targets'], d['preds']))

    # ROCs
    if test_flag is True:
        t = np.array(d['targets'])
        p = np.array(d['preds'])
        fnames = d['fnames']
        io.savemat("predictions/"+model_save_name+"_TEST_"+test_set_name+".mat", dict(fnames=fnames, targets=t, preds=p))

    return running_accuracy.item()/(len(testloader)*config.batch_size)




# class MultiLabelLoss(nn.Module):
#     def __init__(self):
#         super(MultiLabelLoss, self).__init__()

#         self.celoss = nn.CrossEntropyLoss().to(config.device)

#     def forward(self, preds, targets):
#         loss = torch.Tensor([0.])
#         loss = self.celoss()