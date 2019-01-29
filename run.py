import os
import torch
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
from dataloader import *
import argparse
import torch.optim as optim


parser = argparse.ArgumentParser(description='Deep Compression Project')
parser.add_argument('--lr', type=float, default=0.0001, metavar='N',
                    help='learning rate for training (default: 0.001)')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=150, metavar='N',
                    help='number of epochs to train (default: 10)')

# transform = transforms.Compose([transforms.ToTensor()])
data_dir = '/home/sreena/Desktop/Research/Deep Compression/data/'
data_loader = torch.utils.data.DataLoader(Deep_Compression_Dataset(data_dir),batch_size=16, shuffle=True, num_workers=4)

with open(data_dir+'parameters.txt','r') as f:
	for line in f:
		if 'Sparse' in line:
			sparsity = int(line.strip('\r\n').split(':')[1][1:])


A_matrix = []
with open(data_dir + 'matrix.txt','r') as f:
	for line in f:
		A_matrix.append(list(map(float,line.strip('\r\n').split('\t'))))
A_matrix = torch.from_numpy(np.array(A_matrix))



def preprocess(y,x):

	#y :: 16 x100
	#x :: 16 x200
	#A_matrix :: 16x100x200

	r = y - torch.matmul(A_matrix,x)
	out = torch.matmul(A_matrix.transpose(1,2),r)

	return out



for true_x,observ_y in data_loader:

	x = torch.from_numpy(np.zeros((args.batch_size,true_x.shape[1]))).unsqueeze(2)
	observ_y = observ_y.unsqueeze(2)
	true_x = true_x.unsqueeze(2)
	for i in range(sparsity):
		

		inputs = preprocess(observ_y,x)

