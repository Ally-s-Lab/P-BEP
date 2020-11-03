# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 17:53:24 2020

@author: tamiryuv
"""
##############################
import os.path as pth
import yaml
with open('../config.yaml', 'r') as fp:
    config = yaml.load(fp, yaml.FullLoader)
path = pth.dirname(pth.abspath(__file__))[:-3] + '/'
#LinearNet
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler as RU
from sklearn.preprocessing import StandardScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score as AUC_score
from sklearn.model_selection import KFold
import random


def main():
    EPOCHS = 20
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0001

    df = pd.read_csv(path + config['training_data'])

    X = df.iloc[:,:-1].values
    y = df.iloc[:,-1].values

    class trainData(Dataset):

        def __init__(self, X_data, y_data):
            self.X_data = X_data
            self.y_data = y_data

        def __getitem__(self, index):
            return self.X_data[index], self.y_data[index]

        def __len__ (self):
            return len(self.X_data)

    class testData(Dataset):

      def __init__(self,X_test,y_test):
        self.X_test = X_test
        self.y_test = y_test

      def __getitem__(self,index):
        return self.X_test[index], self.y_test[index]

      def __len__(self):
        return len(self.X_test)





    train_data = trainData(torch.FloatTensor(X),
                           torch.FloatTensor(y))

    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## the model architetrure:
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

    # calc accuracy function
    def binary_acc(y_pred, y_test):
        y_pred_tag = torch.round(torch.sigmoid(y_pred))

        correct_results_sum = (y_pred_tag == y_test).sum().float()
        acc = correct_results_sum/y_test.shape[0]
        acc = torch.round(acc * 100)

        return acc


    model = LinearNet()
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(),lr = LEARNING_RATE,momentum = 0.9)
    criterion = torch.nn.CrossEntropyLoss()
    auc_list = []
    kfold = KFold(n_splits = 15)

    total_acc = 0
    for fold, (train_idx,test_idx) in enumerate(kfold.split(X,y)):
      X_train = X[train_idx]
      X_test = X[test_idx]
      y_train = y[train_idx]
      y_test = y[test_idx]

      train_data = trainData(torch.FloatTensor(X_train),
                           torch.LongTensor(y_train))

      train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

      test_data = testData(torch.FloatTensor(X_test),
                           torch.LongTensor(y_test))
      test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

      dataloader = {'train':train_loader,'val': test_loader}

      for epoch in range(EPOCHS):
        print('epoch {} started'.format(epoch))
        for phase in ['train','val']:
          correct = 0
          losses = 0
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



          print('{} Acc : {}, Loss: {}'.format(phase,correct.float() / len(dataloader[phase].dataset), losses.float() / len(dataloader[phase].dataset)))

      total_acc += float(correct*100) / float(BATCH_SIZE*(batch_idx+1))
    total_acc = (total_acc / kfold.get_n_splits())
    print('\n\nTotal accuracy cross validation: {:.3f}%'.format(total_acc))




    checkpoint = {'model': LinearNet(),
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict()}

    torch.save(checkpoint,path + 'models/linear15CV.pth')

if __name__ == '__main__':
    main()