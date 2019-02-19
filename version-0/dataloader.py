
from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import pandas as pd
import os
import numpy as np
from torchvision import datasets, transforms
import torch
import argparse
from random import randint
import random

class Deep_Compression_Dataset(data.Dataset):
	def __init__(self, data_dir,flag):
		super(Deep_Compression_Dataset,self).__init__()
		self.data_dir = data_dir
		with open(data_dir+'parameters.txt','r') as f:
			for line in f:
				if 'Samples' in line:
					self.samples = int(line.strip('\r\n').split(':')[1][1:])
		self.flag = flag

	def __getitem__(self, index):

		obs_txt = self.data_dir + self.flag + '_observations.txt'
		inp_txt = self.data_dir + self.flag + '_sparse_inputs.txt'
		mat_txt = self.data_dir + self.flag + '_matrix.txt'

		with open(obs_txt,'r') as f:
			for i,line in enumerate(f):
				if i==index:
					Y = list(map(float,line.strip('\r\n').split('\t')))

		with open(inp_txt,'r') as f:
			for i,line in enumerate(f):
				if i==index:
					X = list(map(float,line.strip('\r\n').split('\t')))

		with open(mat_txt,'r') as f:
			for i,line in enumerate(f):
				if i==index:
					A = list(map(float,line.strip('\r\n').split('\t')))


		X = torch.from_numpy(np.array(X))
		Y = torch.from_numpy(np.array(Y))
		A = torch.from_numpy(np.array(A).reshape(100,200))

		return X,Y,A

	def __len__(self):

		if self.flag == 'train':

			return int(0.8*(self.samples))
		else:

			return int(0.2*(self.samples))


class DC_Dataset(data.Dataset):
	def __init__(self, sparsity,A_size,num_samples):
		super(DC_Dataset,self).__init__()
		self.sparsity = sparsity
		self.A_size = A_size
		self.num_samples = num_samples

	def __getitem__(self,index):

		A = np.random.randn(self.A_size[0],self.A_size[1])
		A = A/np.linalg.norm(A,axis = 0)

		binary_mask = np.zeros((self.A_size[1],1),dtype = np.uint32);
		rand_indexes = np.arange(self.A_size[1])
		random.shuffle(rand_indexes)
		binary_mask[rand_indexes[0:self.sparsity]] = 1

		X = binary_mask*np.random.randn(self.A_size[1],1)
		X[X==-0] = 0

		Y = np.dot(A,X)

		X = torch.from_numpy(np.array(X.squeeze()))
		Y = torch.from_numpy(np.array(Y.squeeze()))
		A = torch.from_numpy(np.array(A))		

		return X,Y,A

	def __len__(self):

		return self.num_samples





