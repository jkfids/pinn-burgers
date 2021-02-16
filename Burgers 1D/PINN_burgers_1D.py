# -*- coding: utf-8 -*-
"""
@author: Fidel
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import grad
import torch.optim as optim

# Define our neural network via PyTorch
class PINet(nn.Module):
    def __init__(self):
        super(PINet, self).__init__()
        # Takes two input features and outputs a single value
        # 9 hidden layers with a Tanh activation function
        self.inputl1 = nn.Linear(2, 20)
        #self.hidl1 = nn.Linear(20, 20) 
        self.hidl2 = nn.Linear(20, 20) 
        self.hidl3 = nn.Linear(20, 20) 
        self.hidl4 = nn.Linear(20, 20) 
        self.hidl5 = nn.Linear(20, 20) 
        self.hidl6 = nn.Linear(20, 20) 
        self.hidl7 = nn.Linear(20, 20) 
        self.hidl8 = nn.Linear(20, 20) 
        self.hidl9 = nn.Linear(20, 20) 
        self.outl = nn.Linear(20, 1)
        
    # Define the network's forward propagation algorithm
    def forward(self, x):
        activation = nn.Tanh()
        x = activation(self.inputl1(x))
        x = activation(self.hidl2(x))
        x = activation(self.hidl3(x))
        x = activation(self.hidl4(x))
        x = activation(self.hidl5(x))
        x = activation(self.hidl6(x))
        x = activation(self.hidl7(x))
        x = activation(self.hidl8(x))
        x = activation(self.hidl9(x))
        output = activation(self.outl(x))
        return output

# Create the PINN from our Net class
model = PINet()

# Define u(x, t) in Burger's equation as approximated by the PINN
def u_PINN(x, t):
    u = model(torch.stack([x, t]).transpose(0, 1))
    return u

# Define the physics informed regulation function from Burgers' ODE
def f(x, t):
    u = u_PINN(x, t)[0]
    u_t = grad(outputs=u, inputs=t, retain_graph=True)[0]
    u_x = grad(outputs=u, inputs=x, create_graph=True)[0]
    u_xx = grad(outputs=u_x, inputs=x, retain_graph=True)[0]
    f = u_t + u*u_x - (.01/np.pi)*u_xx
    return f

# Import the training data
train = pd.read_csv('output/burgers_1D_train.csv')
target = torch.tensor(train['u(x,t)']).float()
X_train = torch.tensor(train['x'], requires_grad=True).float()
T_train = torch.tensor(train['t'], requires_grad=True).float()
zero = torch.zeros(len(train))

def F():
    F = torch.zeros(len(train))
    for i in range(len(train)):
        F[i] = f(X_train[i].unsqueeze(0), T_train[i].unsqueeze(0))
    return F

def optimize(epochs=100):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=.001)
    for i in range(epochs):
        optimizer.zero_grad()
        output = u_PINN(X_train, T_train).view(1,-1).squeeze(0)
        loss = criterion(output, target)
        print(loss)
        loss.backward()
        optimizer.step()