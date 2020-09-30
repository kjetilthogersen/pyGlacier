#### TEST: Solve diffusion equation and plot
import pyGlacier as solve
import matplotlib.pyplot as plt
import numpy as np

dt = 1
dx = 1
phi = np.ones(100)
sourceTerm = np.zeros(100)
D = np.ones(np.size(phi)+1)
left_boundary_condition = 'Dirichlet'
left_boundary_condition_value = .01;
right_boundary_condition = 'vonNeuman'
right_boundary_condition_value = 0;

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(phi, 'b-')
ax.set_xlim(0, 99)
ax.set_ylim(0, 2)
plt.show()

for i in range(1,1000):
	phi = solve.step_diffusion_1d(dt,dx,phi,sourceTerm,D,left_boundary_condition,left_boundary_condition_value,right_boundary_condition,right_boundary_condition_value)
	line1.set_ydata(phi)
	#fig.canvas.draw()
	plt.pause(0.0001)
	print(i)
