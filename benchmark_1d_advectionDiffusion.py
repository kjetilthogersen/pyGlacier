#### TEST: Solve advection diffusion equation and compare to analytical solution
import pyGlacier as solve
import matplotlib.pyplot as plt
import numpy as np

#phi(x) = c1*D/V*np.exp(-V*x/D) + c2

dt = 1
x = range(-1,1,1000)
dx = x[1]-x[0]
D = 1
V = 1
phi = -np.sin(phi*x*np.exp(-c*x/(2*nu)))

left_boundary_condition = 'Dirichlet'
left_boundary_condition_value = 0.0;
right_boundary_condition = 'Dirichlet'
right_boundary_condition_value = 0.0;

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(phi, 'b-')
ax.set_xlim(0, 99)
ax.set_ylim(0, 2)
plt.show()

for i in range(1,1000):
	phi = solve.step_advection_diffusion_1d(dt = dt,dx = dx, phi = phi, Vphi = V*phi, sourceTerm = np.zeros(np.size(phi)), D = D,left_boundary_condition = left_boundary_condition,left_boundary_condition_value = left_boundary_condition_value, right_boundary_condition = right_boundary_condition, right_boundary_condition_value = right_boundary_condition_value)
	line1.set_ydata(phi)
	#fig.canvas.draw()
	plt.pause(0.0001)
	print(i)
         