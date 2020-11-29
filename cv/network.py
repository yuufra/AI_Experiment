
import torch
import torch.nn as nn
import torch.nn.functional as F

class  MLP(nn.Module):
	"""
	Network / Multilayer perseptron

	"""
	def __init__(self, n_units, n_in, n_out):
		"""
		Construct multilayer perceptron

		Parameters
		----------
		n_units	: int
			dimension of hidden layer
		n_in	: int
			dimension of input
		n_out	: int
			dimension of output
		"""
		super(MLP, self).__init__()

		self.fc1 = nn.Linear(n_in, n_units)	# n_in -> n_units
		self.fc2 = nn.Linear(n_units, n_units)	# n_units -> n_units
		self.fc3 = nn.Linear(n_units, n_out)	# n_units -> n_out

	def forward(self, x):
		"""
		Calculate forward propagation

		Parameters
		----------
		x	: torch.Tensor
			input to the network
			batchsize * input_dimension

		Returns
		-------
		y	: torch.Tensor
			output from the network
			batchsize * outout_dimension
		"""
		h = F.relu(self.fc1(x))
		h = F.relu(self.fc2(h))
		y = self.fc3(h)
		return y

class  MnistCNN(nn.Module):
	"""
	Network / Convolutional newral network for Mnist task
	"""
	def __init__(self, n_out):
		"""
		Construct CNN

		Parameters
		----------
		n_out	: int
			dimension of output
		"""
		super(MnistCNN, self).__init__()
		# Search or See https://pytorch.org/docs/stable/nn.html for information
		# output = (input-kernel)/stride + 1
		# output = floor((input+2*padding-kernel)/stride+1)
		self.conv1 = nn.Conv2d(1,5,5,2,2)	# 1 * 28 * 28 -> 5 * 14 * 14
		self.conv2 = nn.Conv2d(5,10,5,2,2)	# 5 * 14 * 14 -> 10 * 7 * 7
		self.fc = nn.Linear(10 * 7 * 7, n_out)	# 490(=10*7*7) -> n_out

	def forward(self, x):
		"""
		Calculate forward propagation

		Parameters
		----------
		x	: torch.Tensor
			input to the network
			batchsize * channel * height * width

		Returns
		-------
		y	: torch.Tensor
			output from the network
			batchsize * outout_dimension
		"""
		h = F.relu(self.conv1(x))
		h = F.relu(self.conv2(h))
		h = h.view(-1, 10 * 7 * 7)
		y = self.fc(h)
		return y

class   CifarCNN(nn.Module):
	"""
	Network / Convolutional newral network for Cifar10 task
	"""
	def __init__(self, n_out):
		"""
		Construct CNN

		Parameters
		----------
		n_out	: int
			dimension of output
		"""
		super(CifarCNN, self).__init__()

		self.conv1 = nn.Conv2d(3, 6, 5, stride=1)	# 3 * 32 * 32 -> 6 * 28 * 28
		self.pool = nn.MaxPool2d(2, 2)	# 6 * 28 * 28 -> 6 * 14 * 14
		self.conv2 = nn.Conv2d(6, 16, 5, stride=1)	# 6 * 14 * 14 -> 16 * 10 * 10
		#self.pool = nn.MaxPool2d(2, 2)	# 16 * 10 * 10 -> 16 * 5 * 5
		self.fc = nn.Linear(16 * 5 * 5, n_out)	# 400(=16*5*5) -> n_out
		
# 		self.conv1 = nn.Conv2d(3, 6, 5, stride=1)	# 3 * 32 * 32 -> 6 * 28 * 28
# 		self.pool = nn.MaxPool2d(2, 2)	# 6 * 28 * 28 -> 6 * 14 * 14
# 		self.conv2 = nn.Conv2d(6, 16, 5, stride=1)	# 6 * 14 * 14 -> 16 * 10 * 10
# 		#self.pool = nn.MaxPool2d(2, 2)	# 16 * 10 * 10 -> 16 * 5 * 5
# 		self.conv3 = nn.Conv2d(16, 32, 2, stride=1)	# 16 * 5 * 5 -> 32 * 4 * 4
# 		self.fc = nn.Linear(32 * 4 * 4, n_out)	# 400(=16*5*5) -> n_out
		
# 		self.bn1 = nn.BatchNorm2d(6)
# 		self.bn2 = nn.BatchNorm2d(16)
# 		self.bn3 = nn.BatchNorm2d(32)
		
# 		self.dropout = nn.Dropout2d(p=0.7)

	def forward(self, x):
		"""
		Calculate forward propagation

		Parameters
		----------
		x	: torch.Tensor
			input to the network
			batchsize * channel * height * width

		Returns
		-------
		y	: torch.Tensor
			output from the network
			batchsize * outout_dimension
		"""
		h = self.pool(F.relu(self.conv1(x)))
		h = self.pool(F.relu(self.conv2(h)))
		h = h.view(-1, 16 * 5 * 5)
		y = self.fc(h)
		
# 		h = self.pool(self.conv1(x) * torch.sigmoid(self.conv1(x)))
# 		h = self.pool(self.conv2(h) * torch.sigmoid(self.conv2(h)))
		
# 		h = self.pool(F.relu(self.bn1(self.conv1(x))))
# 		h = self.pool(F.relu(self.bn2(self.conv2(h))))
# 		h = F.relu(self.bn3(self.conv3(h)))
# 		h = h.view(-1, 32 * 4 * 4)
# 		y = self.fc(self.dropout(h))

		return y
