# PINN-burgers

Status: WIP

The Navier-Stokes equations are an infamous set of partial differential equations (PDEs) that describe the motion of fluids. The equations can be notoriously difficult to solve and give rise to active areas of research such as turbulence. Burger's equation are a special case formulation of Navier-Stokes that has applications ranging anywhere from fluid mechanics, acoustics, and traffic flow. As such, it is important that we develop methods to simulate these equations.

A major recent development in the field is the introduction of the physics-informed neural network (PINN). PINNs employ AI deep learning to model and simulate non-linear PDEs such as the Burgers equation. It utilizes a special type of neural network that are regularized by physical laws (hence physics-informed) that has numerable advantages over traditional numerical simulation methods.

This repository implements PINNs to solve Burgers' equation, primarily using the PyTorch machine learning library.

Source(s):
https://maziarraissi.github.io/PINNs/
