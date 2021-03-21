# -*- coding: utf-8 -*-
"""
@author: Fidel

(WORK IN PROGRESS)

Implements a physics-informed neural network (PINN) to simulate Burgers
equation in 1D. Code should technically work for large enough f(x,t)
collocation points, however methods to calculate F(x,t) are unoptimized so
optimization can take a long time. Possible solutions include:
    Mini-batch gradient descent rather than batch
    Vectorizatized calculation of F(x,t) to effectively utilize cuda cores
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import grad
import torch.optim as optim

# Import the training data and initialize tensors
col = 10 # No. of collocation points for f(x,t). Ideally col > 2000
train = pd.read_csv('data/train_1D.csv')
target = torch.tensor(train['u(x,t)']).float()
X_train = torch.tensor(train['x']).float()
T_train = torch.tensor(train['t']).float()
X_train.requires_grad=True
T_train.requires_grad=True

X_f = (torch.rand(col) - 0.5)*2
T_f = torch.rand(col)
X_f.requires_grad=True
T_f.requires_grad=True

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define our neural network via PyTorch
class PhysicsInformedNet(nn.Module):
    def __init__(self):
        super(PhysicsInformedNet, self).__init__()
        # Takes two input features and outputs a single value
        # 9 hidden layers with a Tanh activation function
        self.inputl1 = nn.Linear(2, 20)
        self.hidl1 = nn.Linear(20, 20) 
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
        output = self.outl(x)
        return output

# Create the PINN from our PINet class
model = PhysicsInformedNet()

# Define u(x, t) in Burger's equation as approximated by the PINN
def u_PINN(x, t):
    u = model(torch.stack([x, t]).transpose(0, 1))
    return u

# Define the physics-informed regulation function from Burgers' ODE
def f(x, t):
    u = u_PINN(x, t)[0]
    u_t = grad(outputs=u, inputs=t, create_graph=True)[0]
    u_x = grad(outputs=u, inputs=x, create_graph=True)[0]
    u_xx = grad(outputs=u_x, inputs=x, create_graph=True)[0]
    f = u_t + u*u_x - (.01/np.pi)*u_xx
    return f

def F(col=col):
    F = torch.zeros(col)
    for i in range(col):
        #X_f = (torch.rand(col) - 0.5)*2
        #T_f = torch.rand(col)
        #X_f.requires_grad=True
        #T_f.requires_grad=True
        F[i] = f(X_f[i].unsqueeze(0), T_f[i].unsqueeze(0))
    return F

# Optimize the neural network by minising the cost function through batch
# gradient descent algorithms
def optimize(epochs=20000, track=True):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    for i in range(epochs):
        optimizer.zero_grad()
        output = u_PINN(X_train, T_train).view(1,-1).squeeze(0)
        MSE_u = criterion(output, target)
        MSE_f = torch.mean(F()**2)
        loss = MSE_u + MSE_f
        l, u, f = loss.item(), MSE_u.item(), MSE_f.item()
        if track:
            print("MSE: %.6f | MSE_u: %.6f | MSE_f: %.6f" % (l, u, f))
            #print(loss)
        loss.backward()
        optimizer.step()
        
# Using the PINN, plot u(x,t) for t1, t2, t3 over the space of X = [-1,1]
def plot_PINN(t1=0, t2=0.25, t3=0.5, n_x = 201):
    X_tens = torch.linspace(-1, 1, n_x)
    t1_tens = torch.ones_like(X_tens)*t1
    t2_tens = torch.ones_like(X_tens)*t2
    t3_tens = torch.ones_like(X_tens)*t3
    
    Y1 = u_PINN(X_tens, t1_tens).detach().numpy().squeeze()
    Y2 = u_PINN(X_tens, t2_tens).detach().numpy().squeeze()
    Y3 = u_PINN(X_tens, t3_tens).detach().numpy().squeeze()
    X = X_tens.numpy()
    
    fig = plt.figure()
    plt.plot(X, Y1, label = f't = {t1}', linewidth=1)
    plt.plot(X, Y2, label = f't = {t2}', linewidth=1)
    plt.plot(X, Y3, label = f't = {t3}', linewidth=1)
    plt.ylim(-1.25,1.25)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title("PINN simulation of Burgers' Equation in 1D")
    fig.savefig("output/PINN_plot_1D.png")
    plt.close