import torch
from torch import nn
from torch.nn.functional import softmax
class X_vector(nn.Module):
	def __init__(self):
		super(X_vector,self).__init__()
		node_num = [256,512,1024,512,256]
		self.TDNN = nn.Sequential(TDNN(20,node_num[0],5,1),
			TDNN(node_num[0],node_num[1],3,2),
			TDNN(node_num[1],node_num[2],3,3),)
		self.sts_pooling = statistic_pooling()
		self.fc1 = nn.Sequential(fc(node_num[2]*2,node_num[3]),fc(node_num[3],node_num[4]))
		self.fc2 = nn.Linear(node_num[4], 2)
	def forward(self,x):
		x = self.TDNN(x)
		x = self.sts_pooling(x)
		x = self.fc1(x)
		x = softmax(self.fc2(x))
		return x

class TDNN(nn.Module):
	def __init__(self,input_dim,output_dim,kernal_size,dilation):
		super(TDNN,self).__init__()
		self.TDNN = nn.Sequential(nn.Conv1d(input_dim, output_dim, kernal_size, stride=1, dilation=dilation),
			nn.BatchNorm1d(output_dim),nn.ReLU())
	def forward(self,x):
		return self.TDNN(x)
class fc(nn.Module):
	def __init__(self,input_dim,output_dim):
		super(fc,self).__init__()
		self.fc = nn.Sequential(nn.Linear(input_dim, output_dim),nn.BatchNorm1d(output_dim),nn.ReLU())
	def forward(self,x):
		return self.fc(x)
class statistic_pooling(nn.Module):
	def __init__(self):
		super(statistic_pooling,self).__init__()
	def forward(self,x):
		mean_x = x.mean(dim = 2)
		std_x = x.std(dim = 2)
		mean_std = torch.cat((mean_x, std_x), 1)
		return mean_std

