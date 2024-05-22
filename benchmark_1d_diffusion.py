#### TEST: Solve diffusion equation and plot
import pyGlacier.solvers.oneDimensionalSolvers as solve
import matplotlib.pyplot as plt
import numpy as np

dt = 1
x = np.linspace(0,100,100)
dx = x[1]
phi = np.ones(100)
sourceTerm = np.zeros(100)
D = np.ones(np.size(phi)+1)
left_boundary_condition = 'vonNeuman'
left_boundary_condition_value = -.1;
right_boundary_condition = 'Dirichlet'
right_boundary_condition_value = 0.1;

plt.figure()
plt.plot(phi)

for i in range(1,10):
    for i in range(1,200):
        phi = solve.step_diffusion_1d(dt,dx,phi,sourceTerm,D,left_boundary_condition,left_boundary_condition_value,right_boundary_condition,right_boundary_condition_value)
    plt.plot(phi)