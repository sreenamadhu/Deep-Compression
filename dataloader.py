
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


class Deep_Compression_Dataset(data.Dataset):
	def __init__(self, data_dir):
		super(Deep_Compression_Dataset,self).__init__()
		self.data_dir = data_dir
		with open(data_dir+'parameters.txt','r') as f:
			for line in f:
				if 'Samples' in line:
					self.samples = int(line.strip('\r\n').split(':')[1][1:])

		self.obs_txt = data_dir+'observations.txt'
		self.inp_txt = data_dir+'sparse_inputs.txt'

		self.mat_txt = data_dir+'matrix.txt'
		self.matrix = []
		with open(self.mat_txt,'r') as f:
			for line in f:
				self.matrix.append(list(map(float,line.strip('\r\n').split('\t'))))
		self.matrix = np.array(self.matrix)




	def __getitem__(self, index):

		with open(self.obs_txt,'r') as f:
			for i,line in enumerate(f):
				if i==index:
					Y = list(map(float,line.strip('\r\n').split('\t')))

		with open(self.inp_txt,'r') as f:
			for i,line in enumerate(f):
				if i==index:
					X = list(map(float,line.strip('\r\n').split('\t')))

		X = torch.from_numpy(np.array(X))
		Y = torch.from_numpy(np.array(Y))
		A = torch.from_numpy(self.matrix)

		return X,Y,A

	def __len__(self):

		return self.samples