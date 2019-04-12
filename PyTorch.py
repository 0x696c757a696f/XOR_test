import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch import FloatTensor
from torch import optim

use_cuda = torch.cuda.is_available()

X = xor_input = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = xor_output = np.array([[0,1,1,0]]).T

# Converting the X to PyTorch-able data structure.
X_pt = Variable(FloatTensor(X))
X_pt = X_pt.cuda() if use_cuda else X_pt

# Converting the Y to PyTorch-able data structure.
Y_pt = Variable(FloatTensor(Y), requires_grad=False)
Y_pt = Y_pt.cuda() if use_cuda else Y_pt

input_dim = 2
hidden_dim = 8
output_dim = 1

model = nn.Sequential(nn.Linear(input_dim, hidden_dim),    
                      nn.ReLU(),       
                      nn.Linear(hidden_dim, output_dim),
                      nn.Sigmoid())
if use_cuda:
  model.cuda()

criterion = nn.BCELoss()
#criterion = nn.L1Loss()
learning_rate = 0.001
#optimizer = optim.SGD(model.parameters(), lr=learning_rate)
optimizer = optim.Adam(model.parameters())
num_epochs = 10000


for epoch in range(num_epochs):
    predictions = model(X_pt)
    loss_this_epoch = criterion(predictions, Y_pt)
    model.zero_grad()
    loss_this_epoch.backward()
    optimizer.step()
    if epoch%1000 == 0: 
        print("Epoch: {0}, Loss: {1}, ".format(epoch, loss_this_epoch.item()))
print("Function after training:")
print("f(0,0) = {}".format(model1(Variable(torch.FloatTensor([0.0,0.0]).cuda().unsqueeze(0)))))
print("f(0,1) = {}".format(model1(Variable(torch.FloatTensor([0.0,1.0]).cuda().unsqueeze(0)))))
print("f(1,0) = {}".format(model1(Variable(torch.FloatTensor([1.0,0.0]).cuda().unsqueeze(0)))))
print("f(1,1) = {}".format(model1(Variable(torch.FloatTensor([1.0,1.0]).cuda().unsqueeze(0)))))
