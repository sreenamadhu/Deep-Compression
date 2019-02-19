import os
import torch
import torch.utils.data
import numpy as np
from dataloader import *
import argparse
from model import *
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Deep Compression Project')
parser.add_argument('--lr', type=float, default=0.0001, metavar='N',
					help='learning rate for training (default: 0.001)')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
					help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
					help='number of epochs to train (default: 10)')
args = parser.parse_args()

data_dir = '/home/sreena/Desktop/Research/Deep Compression/data/'


fd = open('Sparsity_exp_new.txt','w+')
out = []
for s in range(85,101):

	data_loader = {'train' : torch.utils.data.DataLoader(DC_Dataset(sparsity = s,A_size = [100,200],num_samples = 10000),batch_size=16, shuffle=True, num_workers=4),
				'valid' : torch.utils.data.DataLoader(DC_Dataset(sparsity = s,A_size = [100,200],num_samples = 1000),batch_size=16, shuffle=False, num_workers=4),
				'test' : torch.utils.data.DataLoader(DC_Dataset(sparsity = s, A_size = [100,200], num_samples = 5000), batch_size = 1, shuffle = False, num_workers = 4)}

	net = FC_Model(s)
	t_net = train(net,data_loader,s,args.epochs,result_dir = 'results/')
	score = test(t_net,data_loader,s)
	out.append(score)
	print(score)
	fd.write('Sparsity : {}\tTest Loss : {}\n'.format(s,score))

out = np.array(out)
plt.plot(range(100),out)
plt.title('Test Loss vs Sparsity')
plt.xlabel('sparsity')
plt.ylabel('Test Loss')
plt.show()
