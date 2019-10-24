#### TEST: Solves SIA for an ice sheet with no sliding, and compares it to the 1D solution by Buler et al ("Exact time-depentent similarity solutions for isothermal shallow ice sheets")
import onedimensional_solvers as solve
import matplotlib.pyplot as plt
import numpy as np

# Initialize flowline glacier
x = np.linspace(0,3.5e4,1000)
dx = x[1]-x[0]
#b = 2000.0*np.exp(-x/6700.0)+500
#b = 1250.0 - 1250.0*np.tanh((x-5000.0)/10000.0)
b = 2500.0*np.exp(-x**2.0/10000**2.0)
b = b*0.6-x/30.0+1000.0
#b[-1] = b[-2]
#b[0] = b[1]
H = 0.0*x# + (1-(x-6500)**2.0+10000**2.0)*3e-6
#H = (1-(x-8000)**2.0+20000**2.0)*1e-6*0.5
H[np.where(H<=0)]=0.0


t = 0.0
dt = 1.0e4
SMB = (b-1000.0)*2e-3/(np.pi*1.0e7)*2.50
meltwater_source_term = 1.0e-8
#FrictionLaw = solve.HardBed_RSF(state_parameter = np.ones(np.size(b)), As = 1.0e-24, m = 3.0, q = 2.5, C = 0.4, tc = 1.0e7, dc = 1.0)
FrictionLaw = solve.HardBed_RSF(state_parameter = np.ones(np.size(b)), As = 1.0e-24, m = 3.0, q = 2.5, C = 0.4, tc = 1.0e7, dc = 1.0)
#DrainageSystem = solve.CavitySheet(water_density = 1000.0, hydraulic_potential = 1000.0*9.8*b, background_conductivity = 1.0e-12, source_term = meltwater_source_term, sheet_conductivity = 1e-9, percolation_threshold = 0.4, h0 = 0.1, ev = 1.0e-3)
DrainageSystem = solve.CoupledConduitCavitySheet(hydraulic_potential = 1000.0*9.8*b, water_viscosity = 1.0e-3, latent_heat = 3.0e5,
	source_term = meltwater_source_term, water_density = 1000.0, minimum_drainage_thickness = 1.0, S = 0*b, ev = 1.0e-3,
	percolation_threshold = 0.4, geothermal_heat_flux = 0.0, h0 = 0.01, background_conductivity = 1.0e-11, sheet_conductivity = 1.0e-5,
	channel_constant = 0.001, conduit_spacing = 100.0)

Output = solve.Output(filename = 'test7/benchmarkCoupledSIASSA', output_interval = 100, run_script = 'benchmark_coupledSIA-SSA.py')
glacier = solve.Flowline(width = 1.5e3, A = 1.0e-24, rho = 900.0, n = 3.0, g = 9.8, H = H, b = b, dx = dx, dt = dt, SMB = SMB,
	solver = 'coupled', FrictionLaw = FrictionLaw, DrainageSystem = DrainageSystem, Output = Output)




# Set up figure
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(411)
line1, = ax.plot(x,x*0.0, 'b-', label='Numerical solution')
line2, = ax.plot(x,x*0.0, 'r-', label='Numerical solution')
ax.set_xlim(np.min(x), np.max(x))
ax.set_ylim(0,3e3)
plt.xlabel('x [m]')
plt.ylabel('y [m]')

ax2 = fig.add_subplot(412)
line3, = ax2.plot(x,x*0.0, 'r-', label='water pressure')
line3_2, = ax2.plot(x,x*0.0, 'k-', label='water pressure')
ax2.set_xlim(np.min(x), np.max(x))
ax2.set_ylim(-1e5,1e8)
plt.xlabel('x [m]')
plt.ylabel('water pressure [Pa]')

ax3 = fig.add_subplot(413)
line4, = ax3.plot(x,x*0.0, 'b-', label='theta')
ax3.set_xlim(np.min(x), np.max(x))
ax3.set_ylim(-.01,1.1)
plt.xlabel('x [m]')
plt.ylabel('theta')

ax4 = fig.add_subplot(414)
line5, = ax4.plot(x,x*0.0, 'b-', label='u ssa')
line6, = ax4.plot(x,x*0.0, 'r-', label='u sia')
line7, = ax4.plot(x,x*0.0, 'k-', label='u')
ax4.set_xlim(np.min(x), np.max(x))
ax4.set_ylim(-1e-4,1e-3)
plt.xlabel('x [m]')
plt.ylabel('v [m/s]')

plt.show()

# Time loop
for i in range(int(3.15e6)):


	
	#if (glacier.t>0.6*3.15e7):
#		glacier.dt = 1.0e2
	#if (glacier.t>0.46*3.15e7):
#		glacier.dt = 1.0e2
	#glacier.DrainageSystem.source_term_from_sheet = 0.5*(np.cos(glacier.t*2*3.14/3.15e7)+1.0)
	#glacier.FrictionLaw.state_parameter = np.ones(np.size(b))*0.99

	glacier.DrainageSystem.source_term = meltwater_source_term*(np.sin(glacier.t*3.14/3.15e7)**2.0)

	glacier.step()
	
	#glacier.dt = np.min([np.hstack([dt,glacier.dx/np.max(glacier.u_SSA)/100.0])])
	#glacier.dt = 1.0e1/np.max(np.hstack([glacier.DrainageSystem.S,1e-3]))
	
	#t = t+dt
	#glacier.DrainageSystem.source_term = 2*meltwater_source_term*0.5*(1+np.sin(t/3.15e7*2*np.pi))

	if False:#if(i%10==0): # Plot
		line1.set_ydata(b)
		line2.set_ydata(glacier.H+b)
		line3.set_ydata(glacier.DrainageSystem.water_pressure)
		ax2.set_ylim(np.min(glacier.DrainageSystem.water_pressure)*0.9,np.max(glacier.DrainageSystem.water_pressure)*1.1)

		line3_2.set_ydata(glacier.DrainageSystem.water_pressure)
		
		#line4.set_ydata(glacier.FrictionLaw.state_parameter)

		line4.set_ydata(glacier.FrictionLaw.state_parameter)

		ax3.set_ylim(0,np.max(glacier.FrictionLaw.state_parameter)*1.1)

		#print glacier.DrainageSystem.water_pressure_conduit
		#line5.set_ydata(glacier.u_SSA)
		#line6.set_ydata(glacier.u_SIA)
		line7.set_ydata(glacier.U)
		ax4.set_ylim(np.min(glacier.U)*1.1,np.max(glacier.U)*1.1)
		fig.canvas.draw()
		plt.pause(1.0e-15)
	if(i%1000==0):
		print(str(glacier.t/3.15e7) + ' years')
		#print glacier.DrainageSystem.sheet_source_constant*(glacier.DrainageSystem.water_pressure-glacier.DrainageSystem.sheet_water_pressure)
