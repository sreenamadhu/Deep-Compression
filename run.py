import os
import torch
import torch.utils.data
import numpy as np
from dataloader import *
import argparse
from model import *




parser = argparse.ArgumentParser(description='Deep Compression Project')
parser.add_argument('--lr', type=float, default=0.0001, metavar='N',
					help='learning rate for training (default: 0.001)')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
					help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=150, metavar='N',
					help='number of epochs to train (default: 10)')
args = parser.parse_args()

data_dir = '/home/sreena/Desktop/Research/Deep Compression/data/'
data_loader = {'train' : torch.utils.data.DataLoader(Deep_Compression_Dataset(data_dir,'train'),batch_size=16, shuffle=True, num_workers=4),
				'valid' : torch.utils.data.DataLoader(Deep_Compression_Dataset(data_dir,'valid'),batch_size=16, shuffle=True, num_workers=4)}
with open(data_dir+'parameters.txt','r') as f:
	for line in f:
		if 'Sparse' in line:
			sparsity = int(line.strip('\r\n').split(':')[1][1:])




net = FC_Model(sparsity)
# net.loadweight_from('results/best_model_3.538015625.pth')

train(net,data_loader,sparsity,args.epochs,result_dir = 'results/')

# test(net,A_matrix,data_loader,sparsity)

