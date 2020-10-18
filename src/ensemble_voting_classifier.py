# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 13:33:12 2020

@author: tamiryuv
"""
import torch
import xgboost as xgb
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score,f1_score,roc_curve, auc, classification_report,precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

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




################################################################
# FC_NN Load
#
################################################################
checkpoint3 = torch.load('C:/Users/tamiryuv/Desktop/Research/GuySure_Project/linear15CV.pth', map_location='cpu')
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
checkpoint2 = torch.load('C:/Users/tamiryuv/Desktop/Research/GuySure_Project/LSTM_try.pth',map_location='cpu')
cnn = simple_CNN(in_channels,out_channels,num_classes)
cnn.load_state_dict(checkpoint2['state_dict'])
cnn.eval()
################################################################
# XGB Load
#
################################################################

xgb_model = xgb.Booster()
xgb_model.load_model("../models/xgb_model_al_data2.model")

# Test set : 
positives = pd.read_csv('C:/Users/tamiryuv/Desktop/Research/GuySure_Project/Amitai_sites_df.csv', sep = '\t')
negatives = pd.read_csv('C:/Users/tamiryuv/Desktop/Research/GuySure_Project/01freq_SNPS.csv')
test_data = pd.concat([positives,negatives], axis = 0)
test_data = test_data.sample(5332)
X = test_data.iloc[:,:-1]
y = test_data.iloc[:,-1]
y_tensor = torch.LongTensor(y.values)

# get all predictions for meta_learner :

softmax = torch.nn.Softmax()

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
        if current_pred >= 0.55:
            acc_preds.append(1)
        else:
            acc_preds.append(0)
    metas = np.vstack((meta,for_meta))
    acc = (sum([1 if i==j else 0 for i,j in zip(acc_preds,y.values)])) / len(acc_preds)
    pred_proba = [i for i in roc_auc]
    return acc_preds, acc, pred_proba, metas

y_pred,acc,pred_proba,meta = voting_class(X)

auc_major_votes = roc_auc_score(y_true = y.values, y_score = pred_proba)

################################################################################
class ensemble_features(nn.Module):
  def __init__(self):
    super(ensemble_features,self).__init__()
    
    self.net = nn.Sequential(
    
            nn.Linear(3, 200),
            nn.ReLU(),
            nn.BatchNorm1d(200),
            nn.Linear(200,20),
            nn.ReLU(),
            nn.BatchNorm1d(20),
            nn.Linear(20,2))
            
  def forward(self,x):
      return self.net(x)

  
class train_dataset(Dataset):
    def __init__(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def __len__(self):
        return len(self.X_train)
    
    def __getitem__(self,index):
        inputs = self.X_train[index,:]
        labels = self.y_train[index]
        
        return (inputs, labels)
    

model = ensemble_features()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.0005)        

meta_train = torch.from_numpy(meta).float()


kfold = KFold(n_splits=10)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
 
BATCH_SIZE = 64
EPOCHS = 100
total_acc = 0
CV_auc = []
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots()
for fold, (train_idx,test_idx) in enumerate(kfold.split(meta_train)):
  X_train = meta_train[train_idx]
  X_test = meta_train[test_idx]
  y_train = y_tensor[train_idx]
  y_test = y_tensor[test_idx]

  train_data = train_dataset(torch.FloatTensor(X_test), 
                       torch.LongTensor(y_test))

  train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

  test_data = train_dataset(torch.FloatTensor(X_test),
                       torch.LongTensor(y_test))
  test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

  dataloader = {'train':train_loader,'val': test_loader}

  for epoch in range(EPOCHS):
    
    #print('epoch {} started'.format(epoch))
    for phase in ['train','val']:
      correct = 0
      losses = 0
      auc_l = []
      best_ac = 0
      batch_preds = []
      
      if phase == 'train':
        model.train()
      else:
        model.eval()
      for batch_idx, (x_batch,y_batch) in enumerate(dataloader[phase]):
        x_batch,y_batch = x_batch.to(device), y_batch.to(device)
        outs = model(x_batch)
        loss = criterion(outs,y_batch)
        if phase == 'train':
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
        losses += loss.data
        preds = torch.max(outs.data,dim = 1)[1]
        correct += (preds == y_batch).sum()
        if phase == 'val':
            for_ac = torch.sigmoid(outs[:,1]).data
            batch_auc = roc_auc_score(y_batch,for_ac)
            auc_l.append(batch_auc)
            batch_preds.append(for_ac)
        batch_pr = [i.tolist() for i in batch_preds]
        flat_list = [item for sublist in batch_pr for item in sublist]
    auc_scores = roc_auc_score(y_test, flat_list)
    fpr, tpr, thresholds = roc_curve(y_test, flat_list)
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(auc_scores)
    epoch_auc = np.mean(auc_l)
    if epoch_auc > best_ac:
        best_ac = epoch_auc
        best_wts = model.state_dict()
    CV_auc.append(epoch_auc)
    
        
  
  total_acc += float(correct*100) / float(BATCH_SIZE*(batch_idx+1))
total_auc = np.mean(CV_auc)
total_acc = (total_acc / kfold.get_n_splits()) 
model.load_state_dict(best_wts) 

ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="Receiver operating characteristic (10 fold CV)")
ax.legend(loc="lower right")
plt.show()
print('\n\nTotal accuracy cross validation: {:.3f}% and AUC: {} '.format(total_acc,total_auc))


# print('\n')
# print('Finale Acc Score : {} \nChosen By {}'.format(acc if acc > total_acc else total_acc, 'Majority' if acc > total_acc else 'Meta NN'))


# import shap
# import numpy as np
# import pandas as pd



# explainer = shap.DeepExplainer(model,meta_train)
# shap_values = explainer.shap_values(meta_train)
# shap.summary_plot(shap_values[0], meta_train, plot_type = 'dot', show = False, plot_size = (20,10))
# plt.tight_layout()
# plt.savefig('Meta_SHAP.jpeg')




















