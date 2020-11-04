# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 13:33:12 2020

@author: tamiryuv
"""
import os.path as pth
import yaml
path = pth.dirname(pth.abspath(__file__))[:-3] + '/'
with open(path + 'config.yaml', 'r') as fp:
    config = yaml.load(fp, yaml.FullLoader)

import torch
import xgboost as xgb
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class LinearNet(torch.nn.Module):
    def __init__(self):
        super(LinearNet,self).__init__()
        self.linear1 = torch.nn.Linear(51, 500)
        self.linear2 = torch.nn.Linear(500, 150)
        self.linear3 = torch.nn.Linear(150,20)
        self.linear4 = torch.nn.Linear(20,2)
        
        self.relu = torch.nn.ReLU()
        self.Drop = torch.nn.Dropout(p = 0.2)
        self.batchnorm1 = torch.nn.BatchNorm1d(500)
        self.batchnorm2 = torch.nn.BatchNorm1d(150)
        self.batchnorm3 = torch.nn.BatchNorm1d(20)
    
    def forward(self,x):
        x = self.relu(self.linear1(x))
        x = self.batchnorm1(x)
        x = self.Drop(self.relu(self.linear2(x)))
        x = self.batchnorm2(x)
        x = self.relu(self.linear3(x))
        x = self.batchnorm3(x)
        x = self.linear4(x)
        return x
    
    
class simple_CNN(nn.Module):
  def __init__(self,in_channels,out_channels,num_classes):
    super(simple_CNN,self).__init__()

    self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(1,3))
    self.conv2 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(1,3), padding = 1)
    self.fc1 = nn.Linear(2352,150)
    self.fc2 = nn.Linear(150,num_classes)

    self.bnorm = nn.BatchNorm1d(150)
    
  def forward(self,x):
    out = F.relu(self.conv1(x))
    out = F.relu(self.conv2(out))
    out = out.flatten(start_dim = 1)
    out = F.relu(self.fc1(out))
    out = self.bnorm(out)
    pred = self.fc2(out)
    return pred



def main():
    ################################################################
    # FC_NN Load
    #
    ################################################################
    checkpoint3 = torch.load(path + config['FCNN'], map_location='cpu')
    fc_nn = checkpoint3['model']
    fc_nn.load_state_dict(checkpoint3['state_dict'])
    fc_nn.eval()
    ################################################################
    # CNN Load
    #
    ################################################################
    in_channels = 1
    out_channels = 8
    num_classes = 2
    checkpoint2 = torch.load(path + config['CNN'],map_location='cpu')
    cnn = simple_CNN(in_channels,out_channels,num_classes)
    cnn.load_state_dict(checkpoint2['state_dict'])
    cnn.eval()
    ################################################################
    # XGB Load
    #
    ################################################################

    xgb_model = xgb.Booster()
    xgb_model.load_model(path + config['XGB'])

    # Test set :
    positives = pd.read_csv(path + config['testing_data_positives'])
    negatives = pd.read_csv(path + config['testing_data_negatives'])
    test_data = pd.concat([positives,negatives], axis = 0)
    test_data = test_data.sample(len(test_data))
    X = test_data.iloc[:,:-1]
    y = test_data.iloc[:,-1]
    y_tensor = torch.LongTensor(y.values)

    # get all predictions for meta_learner :

    softmax = torch.nn.Softmax(dim = 1)
    def voting_class(X_test):
        if isinstance(X_test,pd.DataFrame):
            X_test = X_test.values
        acc_preds = []
        roc_auc = []
        for_meta = []
        meta = np.empty((0,3))
        for sample in X_test:
            sample = sample.reshape(1,-1)
            xg_data = xgb.DMatrix(sample)
            torch_data = torch.from_numpy(sample).float()
            xg_pred = xgb_model.predict(xg_data)
            fc_nn_pred = softmax(fc_nn(torch_data))[0][1].detach().numpy()
            cnn_pred = softmax(cnn(torch_data.reshape(1,1,1,51)))[0][1].detach().numpy()
            current_pred = np.mean([xg_pred,fc_nn_pred,cnn_pred])[0]
            roc_auc.append(current_pred)
            for_meta.append([xg_pred.item(),fc_nn_pred.item(),cnn_pred.item()])
            if current_pred >= 0.52:
                acc_preds.append(1)
            else:
                acc_preds.append(0)
        metas = np.vstack((meta,for_meta))
        acc = (sum([1 if i==j else 0 for i,j in zip(acc_preds,y.values)])) / len(acc_preds)
        pred_proba = [i for i in roc_auc]
        return acc_preds, acc, pred_proba, metas
    ### the lines below will output the metrices achived via majority vote : acc, and auc.

    y_pred,acc,pred_proba,meta = voting_class(X)

    auc_major_votes = roc_auc_score(y_true = y.values, y_score = pred_proba)

    def plot_roc_curve(fpr, tpr):
        plt.plot(fpr, tpr, color='orange', label='ROC (AUC = {:.3f})'.format(auc_major_votes))
        plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label = 'Chance')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.show()
    y_true = y.values
    y_scores = pred_proba
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    plot_roc_curve(fpr, tpr)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    pos_pred_val = tp/ (tp+fp)
    neg_pred_val = tn/ (tn+fn)

    # predictions:
    return np.array(y_pred)


if __name__ == '__main__':
    print(main())