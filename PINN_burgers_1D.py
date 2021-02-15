# -*- coding: utf-8 -*-
"""
@author: Fidel
"""
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
import torch.optim as optim

# Define our neural network via PyTorch
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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
PINN_burgers = Net()
# Define the viscosity from Burgers' equation
vis = 0.01/np.pi

# Define u(x, t) in Burger's equation as approximated by the PINN
def uf(x, t):
    u = PINN_burgers(torch.cat([x, t]))
    return u

# Define the physics informed regulation function from Burgers' ODE
def f(x, t):
    u = uf(x, t)[0]
    u_t = grad(outputs=u, inputs=t, retain_graph=True)[0]
    u_x = grad(outputs=u, inputs=x, create_graph=True)[0]
    u_xx = grad(outputs=u_x, inputs=x, retain_graph=True)[0]
    f = u_t + u*u_x - vis*u_xx
    return f

x = torch.ones(1, requires_grad=True)
t = torch.ones(1, requires_grad=True)
PI = f(x, t)

# Calculate the total MSE Loss
output = uf(x, t)
target = torch.ones(1)
criterion = nn.MSELoss()
MSE_u = criterion(output, target) # MSE loss between output and labeled data

MSE_f = f(x, t) # PI regularization term

loss = MSE_u + MSE_f

# Create the L-BFGS optimizer
optimizer = torch.optim.LBFGS(PINN_burgers.parameters())