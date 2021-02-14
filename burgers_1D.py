# -*- coding: utf-8 -*-
"""
@author: Fidel
"""

import numpy as np
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

def u(x, t):
    if t == 0:
        return -np.sin(np.pi*x)
    else:
        I1 = integrate.quad(integrand1, -np.inf, np.inf, args=(x,t))[0]
        I2 = integrate.quad(integrand2, -np.inf, np.inf, args=(x,t))[0]
        return -I1/I2

# Plot u(x,t) for t1, t2, t3 over X
X = np.linspace(-1, 1, n_x)

t1 = 0.25
t2 = 0.5
t3 = 0.75

for i in range(n_x):
    Y1[i] = u(X[i], t1)
    Y2[i] = u(X[i], t2)
    Y3[i] = u(X[i], t3)

fig_static = plt.figure('static')
plt.plot(X, Y1, label = 't = 0.25', linewidth=1)
plt.plot(X, Y2, label = 't = 0.5', linewidth=1)
plt.plot(X, Y3, label = 't = 0.75', linewidth=1)
plt.ylim(-1.25,1.25)
plt.legend()
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.title("Analytical Solution for Burgers' Equation in 1D")
fig_static.savefig("output/burgers_1D.jpg")
plt.close