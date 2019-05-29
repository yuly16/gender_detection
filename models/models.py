'''
@project: Speaker verification SDK
@author: Yutian Li
@file: model_xvector.py
@time: 2019-4-24
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable
from torch.nn.functional import softmax

class X_vector(nn.Module):
    def __init__(self):
        super(X_vector, self).__init__()

        self.tdnn1 = nn.Conv2d(1, 512, (5,20))
        self.bn1 = nn.BatchNorm1d(512)

        self.tdnn2 = nn.Conv2d(1, 512, (5,512))
        self.bn2 = nn.BatchNorm1d(512)

        self.tdnn3 = nn.Conv2d(1, 512, (7,512))
        self.bn3 = nn.BatchNorm1d(512)
 
        self.fc1= nn.Sequential(nn.Linear(1024, 1024),nn.BatchNorm1d(1024),nn.ReLU())
        self.fc2= nn.Sequential(nn.Linear(1024, 512),nn.BatchNorm1d(512),nn.ReLU())
        self.fc3= nn.Linear(512, 2)





    def forward(self, x):

        #frame level tdnn

        x_shape=x.data.shape
        x=x.view(x_shape[0],1,x_shape[1],x_shape[2])
        x = self.tdnn1(x)
        x_shape=x.data.shape
        x=x.view(x_shape[0],x_shape[1],x_shape[2])
        x = F.relu(self.bn1(x))
        x = x.transpose(1, 2).contiguous()

        x_shape=x.data.shape
        x=x.view(x_shape[0],1,x_shape[1],x_shape[2])
        x = self.tdnn2(x)
        x_shape=x.data.shape
        x=x.view(x_shape[0],x_shape[1],x_shape[2])
        x = F.relu(self.bn2(x))
        x = x.transpose(1, 2).contiguous()

        x_shape=x.data.shape
        x=x.view(x_shape[0],1,x_shape[1],x_shape[2])
        x = self.tdnn3(x)
        x_shape=x.data.shape
        x=x.view(x_shape[0],x_shape[1],x_shape[2])
        x = F.relu(self.bn3(x))
        x = x.transpose(1, 2).contiguous()



        mean = torch.mean(x,1)
        std=torch.std(x,1)
        x = torch.cat((mean,std),1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = softmax(self.fc3(x))


        return x