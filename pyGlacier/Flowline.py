import numpy as np
import copy
secondyears = 60*60*24*365.25

from .solvers.oneDimensionalSolvers import *
from .CavitySheet import CavitySheet
from .CoupledConduitCavitySheet import CoupledConduitCavitySheet
from .HardBed_RSF import HardBed_RSF
from .Output import Output

class Flowline:
	#
	# Solves a flowline model based on SIA and/or SIA. Semi-implicit approach for the advection diffusion equation when SSA is active. SIA is solved implicityly. The integration step is performed with Euler.
	# 
	# Dependency on 1D solvers
	#

	def __init__(self, variables):

		self.solver = variables['solver']['ID'] #SIA, SSA, coupled
		self.H = variables['solver']['variables'].get('H')
		self.b = variables['solver']['variables'].get('b')
		self.rho = variables['solver']['variables'].get('rho')
		self.g = variables['solver']['variables'].get('g')
		self.n = variables['solver']['variables'].get('n')
		self.A = variables['solver']['variables'].get('A')
		self.dx = variables['solver']['variables'].get('dx')
		self.dt = variables['solver']['variables'].get('dt')
		self.width = variables['solver']['variables'].get('width')
		self.SMB = variables['solver']['variables'].get('SMB')
		if self.SMB==None:
			self.SMB = zero_func

		# Set up object structure
		if variables['DrainageSystem']['ID']=='CoupledConduitCavitySheet':
			self.DrainageSystem = CoupledConduitCavitySheet(variables = variables['DrainageSystem']['variables'], model = self)
		elif variables['DrainageSystem']['ID']=='CavitySheet':
			raise Exception('CavityShhet implementation not finished')
		elif variables['DrainageSystem']['ID']=='None':
			self.DrainageSystem = None
		else:
			raise Exception('DrainageSystem keyword not recognized')
		if variables['FrictionLaw']['ID']=='HardBed_RSF':
			self.FrictionLaw = HardBed_RSF(variables = variables['FrictionLaw']['variables'], model = self)
		elif variables['FrictionLaw']['ID']=='None':
			self.FrictionLaw = None
		else:
			raise Exception('FrictionLaw keyword not recognized')
		
		self.Output = Output(foldername = variables['Output']['foldername'], output_interval = variables['Output']['output_interval'], flush_interval = variables['Output']['flush_interval'], file_format = variables['Output']['file_format'], model = self, reduced = variables['Output']['reduced'], reduced_output_interval = variables['Output']['reduced_output_interval'])

		self.t = variables['solver']['variables'].get('t')
		self.x = self.dx*np.asarray(range(0,np.size(self.b),1))

		if(self.DrainageSystem is not None):
			self.DrainageSystem.update_water_pressure()

		self.sliding_velocity = np.zeros(np.size(self.b))


		print('********** INITIALIZING FLOWLINE **********')
		print('Solver: ' + variables['solver']['ID'])
		print('FrictionLaw: ' + variables['FrictionLaw']['ID'])
		print('DrainageSystem: ' + variables['DrainageSystem']['ID'])
		print('*******************************************')

	def superposed(self,u):
		yearstoseconds = 365.25*24*60*60
		return 1.0-2.0/np.pi*np.arctan((u/yearstoseconds)**2.0 / 100.0**2.0)

	def step_massContinuity(self):
		# Calculates the mass continuity for a given diffusion coefficients from the
		# the shallow ice approximation and the shallow shelf approximation as well as a given surface mass balance
		h = self.H+self.b
		D_staggered = (self.D[1:]+self.D[0:-1])/2.0 #Staggered grid
		h = step_advection_diffusion_1d(dt = self.dt, dx = self.dx, phi = h, sourceTerm = self.SMB(self), D = D_staggered, Vphi = -self.sliding_velocity*self.H, left_boundary_condition = 'vonNeuman',left_boundary_condition_value = 0.0, right_boundary_condition = 'vonNeuman', right_boundary_condition_value = 0.0)
		#h = step_advection_diffusion_1d(dt = self.dt, dx = self.dx, phi = h, sourceTerm = self.SMB(h), D = D_staggered, Vphi = -self.sliding_velocity*self.H, left_boundary_condition = 'vonNeuman',left_boundary_condition_value = 0.0, right_boundary_condition = 'Dirichlet', right_boundary_condition_value = self.b[-1])
		#h = step_diffusion_1d(dt = self.dt, dx = self.dx, phi = h, sourceTerm = self.SMB, D = D_staggered, left_boundary_condition = 'vonNeuman',left_boundary_condition_value = 0.0, right_boundary_condition = 'vonNeuman', right_boundary_condition_value = 0.0)
		self.H = h-self.b
		self.H[np.where(self.H<=0)] = 0 # Ice thickness cannot be below zero


	def step(self):

		if((self.solver == 'SIA') or (self.solver == 'coupled')):
			self.update_sia_flowline()

		if((self.solver == 'SSA') or (self.solver == 'coupled')):
			self.update_ssa_flowline()
		
		if(self.solver == 'coupled'):
			self.U = (1.0 - self.superposed(self.u_SIA))*self.u_SIA + self.superposed(self.u_SSA)*self.u_SSA
			self.U = self.u_SIA + self.u_SSA
			self.sliding_velocity = self.u_SSA
			self.D = (1.0 - self.superposed(self.u_SIA))*self.D_SIA
			self.D = self.D_SIA

		elif(self.solver == 'SIA'):
			self.U = self.u_SIA 
			self.sliding_velocity = np.zeros(np.size(self.u_SIA))
			self.D = self.D_SIA

		elif(self.solver == 'SSA'):
			self.U = self.u_SSA
			self.sliding_velocity = self.u_SSA
			self.D = np.zeros(np.size(self.D_SSA))
		
		if((self.solver == 'SSA') or (self.solver == 'coupled')):
			self.FrictionLaw.step()
			self.DrainageSystem.step()

		self.step_massContinuity()


		if(self.Output is not None):
			self.Output.save()
		
		self.t = self.t + self.dt

	def runAdaptive(self,t_max,dt_min, dt_max, error_tolerance, interval): #TODO: make sure the error measures will work for all choises of models
		
		print('********** RUN ADAPTIVE **********')
		print('t max: ' + str(t_max/secondyears) + ' years')
		print('dt in [' + str(dt_min) + ',' + str(dt_max) +'] s')
		print('error tolerance: ' + str(error_tolerance))
		print('**********************************')

		j = 0
		i = 0
		rel_error = 1e20
		self.dt = dt_min
		while self.t<t_max:
			i+=1
			
			if(i%interval==0):
				model_copy = copy.deepcopy(self)
				model_copy.Output = None
				model_copy.dt = self.dt/2.0

			self.step()

			if(i%interval==0):
				model_copy.step()
				model_copy.step()
				rel_error_U = np.mean(np.abs(model_copy.U - self.U))/np.max(np.abs(self.U) + 1.0/secondyears)
				rel_error_S = np.mean(np.abs(model_copy.DrainageSystem.S - self.DrainageSystem.S))/np.max(np.mean(self.DrainageSystem.S) + 1.0)
				rel_error = np.max([rel_error_U, rel_error_S])

				if rel_error>error_tolerance:
					self.dt = self.dt/2.0
				elif rel_error<error_tolerance/10:
					self.dt = self.dt*1.1
				if self.dt>dt_max:
					self.dt = dt_max
				elif self.dt<dt_min:
					self.dt = dt_min

			if(self.Output is not None):
				if(i%self.Output.flush_interval==0):
					print(str(self.t/secondyears) + ' years, dt = ' + str(self.dt) + 's, rel_error = ' + str(rel_error))

			if(np.isnan(np.sum(self.U))): # exit simulation if nan values are found in the velocity
				raise Exception('NaN values encountered in soultion. Aborting simulation')

		print('********** RUN ADAPTIVE FINISHED **********')

		return 0

	def run(self,t_max,dt):
		print('********** RUN **********')
		print('t max: ' + str(t_max/secondyears) + ' years')
		print('dt:' + str(dt) + ' s')
		print('**************************')

		i = 0

		self.dt = dt
		if(self.Output is None):
			interval = 100
		else:
			interval = self.Output.output_interval

		while self.t<t_max:
			i+=1
			self.step()
			if(i%interval==0):
				print(str(self.t/secondyears) + ' years, dt = ' + str(self.dt) + 's')

		return 0


	def runInitializeSIA(self,dt,t_max):
		
		print('********** INITIALIZE SIA **********')

		# Initialize thickness with SIA solution (no sliding or drainage)
		model_copy = copy.deepcopy(self)
		model_copy.solver = 'SIA'
		model_copy.DrainageSystem = None
		model_copy.FrictionLaw = None
		model_copy.Output = None
		model_copy.run(t_max = t_max, dt = dt)
		self.H = model_copy.H

		print('********** INITIALIZE FINISHED **********')

		return 0

	def getDictionary(self, init = False):

		if init:
			return {'A':self.A, 'b':self.b.tolist(), 'dx':self.dx, 'rho':self.rho, 'g':self.g, 'n':self.n, 'width':self.width, 'solver':self.solver, 'x':self.x.tolist(), 'dt':self.dt}
		else:
			dictionary = {'U':self.U.tolist(), 'H':self.H.tolist(), 't':self.t, 'length':np.sum(self.H>0)*self.dx}
			if self.solver == 'coupled':
				dictionary.update({'U_SIA':self.u_SIA.tolist(), 'U_SSA':self.u_SSA.tolist()})
			return dictionary

	def update_sia_flowline(self):
		#
		# Updates the "diffusion" coefficient and the ice velocity for the shallow ice approximation
		#
		h = self.H+self.b
		h_bc = np.hstack([h[0],h,h[-1]])
		dhdx = (h_bc[1:]-h_bc[0:-1])/self.dx
		dhdx = (dhdx[1:]+dhdx[0:-1])/2.0
		self.u_SIA = -2.0*(self.rho*self.g)**self.n *self.A*(1.0/(self.n+1.0))*dhdx**self.n *self.H**(self.n+1.0)
		self.D_SIA = 2.0*(self.rho*self.g)**self.n *self.A*(1.0/(self.n+2.0))*dhdx**(self.n-1.0) *self.H**(self.n+2.0)


	def update_ssa_flowline(self,tol=1.0e-3,itermax=500,eps=1.0e-20):
		#
		# Updates the "diffusion" coefficient and the ice velocity for the shallow shelf approximation, the input u is taken as the initial guess
		#
		h = self.H+self.b
		dhdx = np.gradient(h)/self.dx
		driving_stress = self.rho*self.g*self.H*dhdx
		
		rel_error = 1.0
		i = 0

		u = self.sliding_velocity

		while((rel_error>tol)&(i<itermax)):

			self.FrictionLaw.update_friction_coefficient() #Update friction coefficient
			self.FrictionLaw.update_lateral_drag() #Update lateral drag

			alpha = self.FrictionLaw.friction_coefficient + self.FrictionLaw.lateral_drag #Combine basal and lateral drag in a single coefficient

			dudx = (u[1:] - u[0:-1])/self.dx # On staggered grid
			dudx_sqr_reg = dudx**2.0 + eps**2.0
			
			W = 2*self.A**(-1.0/self.n)*(self.H[1:] + self.H[0:-1])/2.0*dudx_sqr_reg**(((1.0/self.n)-1.0)/2.0)
			W = np.hstack([W,W[-1]])
			u_new = step_elliptical_1d(dx = self.dx, W = W, alpha = alpha, beta = driving_stress, gamma = driving_stress*0, left_boundary_condition = 'Dirichlet',left_boundary_condition_value = 0.0, right_boundary_condition = 'Dirichlet', right_boundary_condition_value = 0.0)
			rel_error = np.mean(  np.abs(u_new-u)/(np.abs(u_new)+eps) )

			u = u_new
			i = i+1
		
		if(i==itermax):
			print('warning: maximum number of iterations reached in SSA solver')

		self.u_SSA = u
		self.D_SSA = self.u_SSA*self.H