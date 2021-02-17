# -*- coding: utf-8 -*-
"""
@author: Fidel
"""

import numpy as np
import pandas as pd
from scipy import integrate
from matplotlib import pyplot as plt

# Initialize matrices/constants
vis = .01/np.pi # Viscocity term in Burger's equation
n_x = 201 # Number of u(x, t) elements on x
X = np.zeros(n_x)
Y1 = np.zeros(n_x)
Y2 = np.zeros(n_x)
Y3 = np.zeros(n_x)
Y_ani = np.zeros(n_x)

# Analytical solution for Burger's equation with periodic boundary conditions 
# at x = [-1,1] and initial condition u(x,0) = -sin(pi*x)
def f_cole(y):
    return np.exp(-np.cos(np.pi*y)/(2*np.pi*vis))

def integrand1(eta, x, t):
    return np.sin(np.pi*(x-eta))*f_cole(x-eta)*np.exp(-eta**2/(4*vis*t))

def integrand2(eta, x, t):
    return f_cole(x-eta)*np.exp(-eta**2/(4*vis*t))

def u_anal(x, t):
    if t == 0:
        return -np.sin(np.pi*x)
    else:
        I1 = integrate.quad(integrand1, -np.inf, np.inf, args=(x,t))[0]
        I2 = integrate.quad(integrand2, -np.inf, np.inf, args=(x,t))[0]
        return -I1/I2

# Plot u(x,t) for t1, t2, t3 over X
def plot_anal(t1=0, t2=0.25, t3=0.5, n_x = 201):
    X = np.linspace(-1, 1, n_x)
    
    for i in range(n_x):
        Y1[i] = u_anal(X[i], t1)
        Y2[i] = u_anal(X[i], t2)
        Y3[i] = u_anal(X[i], t3)
    
    fig_static = plt.figure('static')
    plt.plot(X, Y1, label = f't = {t1}', linewidth=1)
    plt.plot(X, Y2, label = f't = {t2}', linewidth=1)
    plt.plot(X, Y3, label = f't = {t3}', linewidth=1)
    plt.ylim(-1.25,1.25)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title("Analytical Solution for Burgers' Equation in 1D")
    fig_static.savefig("output/burgers_1D_anal.png")
    plt.close
    
# Generate training data for PINN and save it to a csv file
def gen_train(half_set, max_t):
    m = int(half_set)
    tr_init = np.zeros([m,3])
    tr_bound = np.zeros([m,3])
    
    tr_init[:,1] = np.linspace(-1, 1, m)
    tr_bound[:,1] = np.ones(m) - 2*np.random.randint(2, size=m)
    tr_bound[:,2] = np.linspace(0, max_t, m)
    for i in range(m):
        tr_init[i][0] = u_anal(tr_init[i][1], 0)
        tr_bound[i][0] = u_anal(tr_bound[i][1], tr_bound[i][2])
    training = np.append(tr_init, tr_bound, axis=0)
    df = pd.DataFrame(training, columns=['u(x,t)', 'x', 't'])
    df.to_csv('output/burgers_1D_train.csv', index=False)