import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def fcn():

	net = []
	net.append(nn.Linear(200,256))
	net.append(nn.ReLU(True))
	net.append(nn.Linear(256,200))

	return nn.Sequential(*net)

class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


class FC_Model(nn.Module):
	def __init__(self,sparsity):
		super(FC_Model, self).__init__()

		model = [fcn() for i in range(sparsity)]
		self.model = ListModule(*model)

	def forward(self,inputs,x,stage):

		inputs = inputs.view(inputs.shape[0],-1)
		inputs = self.model[stage](inputs)
		_,location_x = torch.max(inputs,1)
		index = torch.from_numpy(np.arange(inputs.shape[0]))
		values_x = F.max_pool1d(inputs.unsqueeze(1), kernel_size=inputs.size()[1:])
		clone_x = x.clone()
		clone_x[index,location_x] = values_x.view(-1)
		x = clone_x
		return x

	def loadweight_from(self, pretrain_path):
		pretrained_dict = torch.load(pretrain_path)
		model_dict = self.state_dict()
		pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
		model_dict.update(pretrained_dict)
		self.load_state_dict(model_dict)

	def cls_loss(self, pred_x, true_x):
		loss = torch.sum(torch.abs(pred_x - true_x),1)
		loss = torch.mean(loss) #mean over batch
		return loss

def preprocess(y,x,A):

	#y :: 16 x100
	#x :: 16 x200
	#A_matrix :: 16x100x200

	x = x.unsqueeze(2)

	r = y - torch.matmul(A,x) #16,100,1
	out = torch.matmul(A.transpose(1,2),r) #16,200,1
	return out


def train(net,data_loader,sparsity,epochs,result_dir):

	if torch.cuda.is_available():
		key = "cuda:0"
	else:
		key = "cpu"
	device = torch.device(key)
	net.to(device)
	
	mse_loss = torch.nn.MSELoss(size_average=True)
	optimizer = optim.Adam(filter(lambda p : p.requires_grad, net.parameters()), lr = 0.0001)
	min_loss = float("inf")
	
	fd = open(result_dir + 'logs.txt', 'w+')
	for epoch in range(epochs):
		net.train()
		current_loss = 0.0

		for true_x,observ_y,A_matrix in data_loader['train']:


			A_matrix = A_matrix.to(device).float()

			true_x = true_x.to(device).float()
			observ_y = observ_y.to(device).float()
			x = torch.zeros(true_x.shape[0],true_x.shape[1],requires_grad=True)
			x = x.to(device).float()
			observ_y = observ_y.unsqueeze(2)
			true_x = true_x.unsqueeze(2)
			optimizer.zero_grad()

			for i in range(sparsity):
				inputs = preprocess(observ_y,x,A_matrix)
				x = net(inputs,x,i)

			# norm = torch.sum(true_x.norm(p=2, dim=1, keepdim=True))

			loss = mse_loss(x.unsqueeze(2),true_x)
			loss.backward()
			optimizer.step()
			current_loss = current_loss + loss.item()*true_x.size(0)

		train_loss = current_loss/ float(len(data_loader['train'].dataset))

		net.eval()
		current_loss = 0.0

		for true_x,observ_y,A_matrix in data_loader['valid']:

			A_matrix = A_matrix.to(device).float()
			true_x = true_x.to(device).float()
			observ_y = observ_y.to(device).float()
			x = torch.zeros(true_x.shape[0],true_x.shape[1],requires_grad=False)
			x = x.to(device).float()
			observ_y = observ_y.unsqueeze(2)
			true_x = true_x.unsqueeze(2)
			optimizer.zero_grad()
			for i in range(sparsity):
				inputs = preprocess(observ_y,x,A_matrix)
				x = net(inputs,x,i)
			norm = torch.sum(true_x.norm(p=2, dim=1, keepdim=True))
			loss = mse_loss(x.unsqueeze(2),true_x).div(norm)
			current_loss = current_loss + loss.item()*true_x.size(0)
		valid_loss = current_loss/ float(len(data_loader['valid'].dataset))


		if valid_loss < min_loss:
			min_loss = valid_loss
			torch.save(net.state_dict(),result_dir + 'best_model_s_{}.pth'.format(sparsity))
			best_model_path = result_dir + 'best_model_s_{}.pth'.format(sparsity)

		epoch_info = 'Epoch {} : Train Loss {} \t Valid Loss {} '.format(epoch,train_loss,valid_loss)
		print(epoch_info)
		fd.write(epoch_info)

	net.loadweight_from(best_model_path)
	return net


def test(net,data_loader,sparsity):

	if torch.cuda.is_available():
		key = "cuda:0"
	else:
		key = "cpu"

	device = torch.device(key)
	net.to(device)
	mse_loss = torch.nn.MSELoss(size_average=True)
	net.eval()
	current_loss = 0.0
	optimizer = optim.Adam(filter(lambda p : p.requires_grad, net.parameters()), lr = 0.0001)

	for true_x,observ_y,A_matrix in data_loader['test']:

		A_matrix = A_matrix.to(device).float()
		true_x = true_x.to(device).float()
		observ_y = observ_y.to(device).float()
		x = torch.zeros(true_x.shape[0],true_x.shape[1],requires_grad=False)
		x = x.to(device).float()
		observ_y = observ_y.unsqueeze(2)
		true_x = true_x.unsqueeze(2)
		optimizer.zero_grad()
		for i in range(sparsity):
			inputs = preprocess(observ_y,x,A_matrix)
			x = net(inputs,x,i)
		norm = torch.sum(true_x.norm(p=2, dim=1, keepdim=True))
		loss = mse_loss(x.unsqueeze(2),true_x).div(norm)
		current_loss = current_loss + loss.item()*true_x.size(0)
	valid_loss = current_loss/ float(len(data_loader['test'].dataset))

	return valid_loss
