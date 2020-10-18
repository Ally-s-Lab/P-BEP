# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 17:57:06 2020

@author: tamiryuv
"""
##############################
#CNN :

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

def main():

    df = pd.read_csv('../lib/training.csv')

    X = df.iloc[:,:-1].values
    y = df.iloc[:,-1].values




    def save_checkpoint(state, filename = '../models/CNN2.pth'):
      print('SAVING CHECKPOINT')
      torch.save(state, filename)

    class trainData(Dataset):
      def __init__(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train

      def __len__(self):
        return len(self.X_train)

      def __getitem__(self,index):
        return self.X_train[index], self.y_train[index]


    class simple_CNN(nn.Module):
      def __init__(self,in_channels,out_channels,num_classes):
        super(simple_CNN,self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(1,3))
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(1,3), padding = 1)
        self.fc1 = nn.Linear(4512,150)
        self.fc2 = nn.Linear(150,num_classes)

        self.bnorm = nn.BatchNorm1d(150)
        self.pool = nn.MaxPool2d(kernel_size=(1,3),stride=1)



      def forward(self,x):
        out = F.relu(self.conv1(x))
        out = self.pool(out)
        out = F.relu(self.conv2(out))
        out = out.flatten(start_dim = 1)
        out = F.relu(self.fc1(out))
        out = self.bnorm(out)
        pred = self.fc2(out)
        return pred



    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    in_channels = 1
    out_channels = 16
    num_classes = 2
    kfold = KFold(n_splits=10)
    model = simple_CNN(in_channels,out_channels,num_classes)
    model.to(device)

    optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.01)
    criterion = nn.CrossEntropyLoss()
    BATCH_SIZE = 128
    epochs = 20
    best_model_wts = model.state_dict()
    best_acc = 0.5
    total_acc = []
    for fold,(train_idx,test_idx) in enumerate(kfold.split(X)):
      X_train = X[train_idx]
      y_train = y[train_idx]
      X_test = X[test_idx]
      y_test = y[test_idx]

      train = trainData(X_train,y_train)
      test = trainData(X_test,y_test)

      train_loader = DataLoader(train,batch_size=BATCH_SIZE,shuffle = True)
      test_loader = DataLoader(test,batch_size=BATCH_SIZE)

      dataloader = {'train': train_loader,'val': test_loader}

      for i in range(epochs):
        print('epoch : {}'.format(i))
        for phase in ['train','val']:
          running_loss = 0
          epoch_acc = 0
          for seq, labels in dataloader[phase]:
            if phase == 'train':
              model.train()
            else:
              model.eval()
            seq = seq.unsqueeze(1)
            batch_s = seq.shape[0]
            seq = seq.reshape(batch_s,1,1,51).float()
            seq, labels = seq.to(device), labels.to(device)
            y_pred = model(seq)
            single_loss = criterion(y_pred.squeeze(0), labels.long())
            if phase == 'train':
              optimizer.zero_grad()
              single_loss.backward()
              optimizer.step()
            running_loss += single_loss.item()
            batch_acc = (torch.argmax(y_pred,1) == labels).sum()
            epoch_acc += batch_acc
            if phase == 'val' and epoch_acc > best_acc:
              best_acc = epoch_acc
              checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
             # save_checkpoint(checkpoint)



          print('{} loss: {} \t acc: {}'.format(phase,running_loss / len(dataloader[phase].dataset), epoch_acc.item() / len(dataloader[phase].dataset)))
      total_acc.append(epoch_acc.item() / len(y_test))
    print('\n\nTotal accuracy cross validation: {:.3f}%'.format(np.mean(total_acc)*100))

if __name__ == '__main__':
    main()