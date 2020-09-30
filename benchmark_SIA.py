#### TEST: Solves SIA for an ice sheet with no sliding, and compares it to the 1D solution by Buler et al ("Exact time-depentent similarity solutions for isothermal shallow ice sheets")
import pyGlacier as pg
import matplotlib.pyplot as plt
import numpy as np

def halfar(t,r,gamma): # Returns the analytical solution at a given time
	H1 = 1e5
	n = 3.0
	alpha = 1.0/11.0
	lmbd = ((n+2)*alpha/gamma)**(1.0/n)*(2.0*n+1.0)/(n+1.0)
	theta = H1**((2.0*n+1.0)/(n+1.0))*lmbd**(-n/(n+1.0))
	H = t**(-alpha)*(H1**((2.0*n+1.0)/n) - lmbd*(t**(-alpha)*abs(r))**((n+1.0)/n))**(n/(2.0*n+1.0))
	H[np.where(abs(r)>theta*t**(alpha))]=0
	return H

A = 1e-24 #Rheology constant
rho = 900.0 #Ice density
g = 9.8
n = 3.0
x = np.linspace(-1e6,1e6,1000)
dx = x[1]-x[0]
b = np.zeros(np.size(x))
gamma = A*2.0*(rho*g)**n
t = np.linspace(1,100,100000)
H_analytical = halfar(t[0],x,gamma)
H_numerical = H_analytical


def SMB(height):
	return 0*height


variables = {
'solver': 
	{'ID': 'SIA',
	'variables':{
		'rho': 900.0,
		'g': 9.8,
		'n': 3.0,
		't': 0.0,
		'A': 1.0e-24,
		'H': H_numerical,
		'b': b,
		'dx': dx,
		'dt': t[1]-t[0]} },
'DrainageSystem': 
	{'ID': 'None'},
'FrictionLaw': 
	{'ID': 'None'},
'Output': 
	{'foldername': 'benchmark_SIA',
	'output_interval': 100,
	'file_format': 'json'} }


# Loop and plot
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(x,H_analytical, 'b-', label='Analytical solution')
line2, = ax.plot(x,H_numerical, 'r--', label='Numerical solution')
ax.set_xlim(-1e6, 1e6)
ax.set_ylim(0, 1e5)
plt.xlabel('x [m]')
plt.ylabel('y [m]')
ax.legend()
plt.show()

# Create glacier object
glacier = pg.Flowline(variables = variables)

# Time loop
for i in range(1,np.size(t)):
	glacier.step()
	H_analytical = halfar(t[i],x,gamma) # Analytical solution
	if(i%100==0): # Plot
		print(t[i])
		line1.set_ydata(H_analytical)
		line2.set_ydata(glacier.H)
		fig.canvas.draw()
		plt.pause(0.0001)
