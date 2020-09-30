#
# Written by Kjetil Thøgersen
#
#
#
#

import numpy as np
from scipy.sparse import spdiags
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import dsolve
import scipy.io as sio
import json
import sys
import os
import copy
import matplotlib.pyplot as plt
import math

secondyears = 60*60*24*365.25

def zero_func(model): #Default zero function (if function not supplied by user)
	return model.b*0

def step_diffusion_1d(dt,dx,phi,sourceTerm,D,left_boundary_condition,left_boundary_condition_value,right_boundary_condition,right_boundary_condition_value):
	#
	# Solves 1D diffusion on staggered grid; give the midpotins only, not the artificial points outside the boundary, they are specified by the given boundary condition. 
	# Input values are dt, dx, phi (solution in the previous time-step) on the nodes, source term on the nodes, diffusion coefficient vector on midpoints (size N+1)
	# Boundary conditions can be vonNeuman or Dirichlet
	#
	# @TODO: give equation
	# 
	def boundary_modifier(x,dt):
		return {
        	'Dirichlet': [1,0,1],
        	'vonNeuman': [1,-1,dt]
    	}[x]
	N_intervals = np.size(phi)-1
	rhs = -dx*dx*(phi/dt+sourceTerm)
	left_modifier = boundary_modifier(left_boundary_condition,dt)
	right_modifier = boundary_modifier(right_boundary_condition,dt)
	rhs[0] = -left_modifier[2]*left_boundary_condition_value
	rhs[np.size(rhs)-1] = right_modifier[2]*right_boundary_condition_value
	A = csc_matrix(spdiags( [ np.hstack((D[0:N_intervals-1],right_modifier[1],0)),  np.hstack((left_modifier[0],-(D[0:N_intervals-1]+D[1:N_intervals]+dx*dx/dt),right_modifier[0])),   np.hstack((0,left_modifier[1],D[1:N_intervals]))  ], [-1,0,1], N_intervals+1, N_intervals+1))
	phi = dsolve.spsolve(A, rhs, use_umfpack=True)
	return phi


def step_advection_diffusion_1d(dt,dx,phi,sourceTerm,D,Vphi,left_boundary_condition,left_boundary_condition_value,right_boundary_condition,right_boundary_condition_value):
	#
	# Solves 1D advection diffusion on staggered grid; give the midpotins only, not the artificial points outside the boundary, they are specified by the given boundary condition. 
	# Input values are dt, dx, phi on the nodes, source term on the nodes, diffusion coefficient vector on midpoints (size N+1), as well as advection term Vphi on midpoints
	# Boundary conditions can be vonNeuman or Dirichlet
	#
	# @TODO: give equation
	# 
	def boundary_modifier(x,dt):
		return {
        	'Dirichlet': [1,0,1],
        	'vonNeuman': [1,-1,dt]
    	}[x]
	N_intervals = np.size(phi)-1
	left_modifier = boundary_modifier(left_boundary_condition,dt)
	right_modifier = boundary_modifier(right_boundary_condition,dt)
	rhs = -dx*dx*(phi/dt+sourceTerm + np.gradient(Vphi)/dx) #Here we are treating the advection term explicitly as an rhs contribution
	rhs[0] = -left_modifier[2]*left_boundary_condition_value
	rhs[-1] = right_modifier[2]*right_boundary_condition_value

	A = csc_matrix(spdiags( [ np.hstack((D[0:N_intervals-1],right_modifier[1],0)),  np.hstack((left_modifier[0],-(D[0:N_intervals-1]+D[1:N_intervals]+dx*dx/dt),right_modifier[0])),   np.hstack((0,left_modifier[1],D[1:N_intervals]))], [-1,0,1], N_intervals+1, N_intervals+1))
	phi = dsolve.spsolve(A, rhs, use_umfpack=True)
	return phi

def step_elliptical_1d(dx,W,alpha,beta,gamma,left_boundary_condition,left_boundary_condition_value,right_boundary_condition,right_boundary_condition_value):
	#
	# Solves 1D elliptical equation on staggered grid (@TODO: give equation)
	# (W(x) u_x)_x - alpha(x) u + gamma(x)*u_x = beta(x)
	#
	def boundary_modifier(x,val):
		return {
			'Dirichlet': [val,0,val],
			'vonNeuman': [-val,val,-val]
		}[x]
	N_intervals = np.size(alpha)-1
	val = np.mean(np.abs(2*W+dx**2.0*alpha - gamma*dx))
	left_modifier = boundary_modifier(left_boundary_condition,val)
	right_modifier = boundary_modifier(right_boundary_condition,val)
	rhs = dx*dx*beta
	rhs[0] = left_modifier[2]*left_boundary_condition_value
	rhs[np.size(rhs)-1] = right_modifier[2]*right_boundary_condition_value
	A = csc_matrix(spdiags( [ np.hstack((W[0:N_intervals-1], right_modifier[1],0)),  np.hstack((left_modifier[0],-(W[0:N_intervals-1]+W[1:N_intervals]+alpha[1:N_intervals]*dx*dx) - (gamma[1:N_intervals]+gamma[0:N_intervals-1])/2.0*dx,right_modifier[0])),   np.hstack((0,left_modifier[1],W[1:N_intervals] + (gamma[1:N_intervals]+gamma[0:N_intervals-1])/2.0*dx))  ], [-1,0,1], N_intervals+1, N_intervals+1))
	u = dsolve.spsolve(A, rhs, use_umfpack=True)
	#print A
	return u


class Flowline:
	#
	# Solves a flowline model based on SIA and/or SIA. Semi-implicit approach for the advection diffusion equation when SSA is active. SIA is solved implicityly. The integration step is performed with Euler.
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
		self.Output = Output(foldername = variables['Output']['foldername'], output_interval = variables['Output']['output_interval'], file_format = variables['Output']['file_format'], model = self)

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
					self.dt = self.dt/1.5
				elif rel_error<error_tolerance/10:
					self.dt = self.dt*1.5
				if self.dt>dt_max:
					self.dt = dt_max
				elif self.dt<dt_min:
					self.dt = dt_min

			if(self.Output is not None):
				if(i%self.Output.output_interval==0):
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
			dictionary = {'U':self.U.tolist(), 'H':self.H.tolist(), 't':self.t}
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

class CavitySheet():
	#
	# Solves for the hydraulic potential in a cavity sheet where the sheet hydraulic conductivity is set by the state parameter from the sliding law.
	# 
	def __init__(self,hydraulic_potential = 0.0, ev = 1.0e-3, percolation_threshold = 0.6, geothermal_heat_flux = 0.0, h0 = 1.0, latent_heat = 3.35e5, source_term = zero_func, background_conductivity = 1.0e-15, sheet_conductivity = 1.0e-6, water_density = 1000.0, minimum_drainage_thickness = 1.0, model = None):
		
		self.hydraulic_potential = hydraulic_potential
		self.ev = ev
		self.percolation_threshold = percolation_threshold
		self.geothermal_heat_flux = geothermal_heat_flux
		self.h0 = h0
		self.latent_heat = latent_heat
		self.source_term = source_term
		self.background_conductivity = background_conductivity
		self.sheet_conductivity = sheet_conductivity
		self.water_density = water_density
		self.minimum_drainage_thickness = minimum_drainage_thickness
		self.model = model


	def update_water_pressure(self):
		model = self.model
		self.water_pressure = self.hydraulic_potential - self.water_density*model.g*model.b
		
		#Force water pressure below or equal to zero when glacier thickness vanishes
		ind = np.where( np.logical_and(model.H<=self.minimum_drainage_thickness, self.water_pressure>0.0 ))
		self.water_pressure[ind]=0.0
		self.hydraulic_potential[ind]= self.water_density*model.g*model.b[ind]

	def step(self,source_term_from_conduit = 0.0):
		model = self.model
		dissipation_friction = np.abs(model.sliding_velocity**2.0*model.FrictionLaw.friction_coefficient)
		melt_rate = 1.0/(self.latent_heat*model.rho)*(self.geothermal_heat_flux + dissipation_friction) # Calculate melt rate
		effective_conductivity = self.getHydraulicConductivity()

		D = effective_conductivity*self.water_density*model.g/self.ev
		D_staggered = (D[1:]+D[0:-1])/2.0
		
		#beta = self.water_density*model.g/self.ev*(self.h0*model.FrictionLaw.state_parameter_derivative + self.source_term(model) + melt_rate + source_term_from_conduit)
		beta = self.water_density*model.g/self.ev*(1/model.FrictionLaw.state_parameter*self.h0*model.FrictionLaw.state_parameter_derivative + self.source_term(model) + melt_rate + source_term_from_conduit) # with the logarithmic term
		self.hydraulic_potential = step_diffusion_1d(dt = model.dt, dx = model.dx, phi = self.hydraulic_potential, sourceTerm = beta, D = D_staggered, left_boundary_condition = 'vonNeuman',left_boundary_condition_value = 0.0, right_boundary_condition = 'vonNeuman', right_boundary_condition_value = 0.0)

		self.update_water_pressure()

	def getDischarge(self):
		model = self.model
		effective_conductivity = self.getHydraulicConductivity()
		discharge = - effective_conductivity*np.gradient(self.hydraulic_potential/model.dx)
		return discharge
		
	def getHydraulicConductivity(self):
		model = self.model
		perc_fun = self.percolation_function()
		#effective_conductivity = self.background_conductivity*np.ones(np.size(model.b)) + self.sheet_conductivity*self.h0*(1.0-model.FrictionLaw.state_parameter)**3.0*perc_fun
		effective_conductivity = self.background_conductivity*np.ones(np.size(model.b)) - self.sheet_conductivity*(self.h0*np.log(model.FrictionLaw.state_parameter))**3.0*perc_fun #log-term for opening
		return effective_conductivity

	def percolation_function(self):
		model = self.model
		return .5*(np.tanh((self.percolation_threshold-model.FrictionLaw.state_parameter)*5.0)+1.0)

	def getDictionary(self, init = False):
		model = self.model
		if init:
			return {'ev':self.ev, 'h0':self.h0, 'water_density':self.water_density, 'latent_heat':self.latent_heat, 'minimum_drainage_thickness':self.minimum_drainage_thickness,
			'percolation_threshold':self.percolation_threshold, 'background_conductivity':self.background_conductivity, 'geothermal_heat_flux':self.geothermal_heat_flux,
			'sheet_conductivity':self.sheet_conductivity, 'channel_constant':self.channel_constant, 'closure_coefficient':self.closure_coefficient}
		else:
			return {'hydraulic_potential':self.hydraulic_potential.tolist(), 'water_pressure':self.water_pressure.tolist(), 'SourceTerm':self.source_term(model)}

class CoupledConduitCavitySheet():
	def __init__(self, variables, model):
		#
		# Sets up a coupled cavity conduit drainage system
		#
		self.hydraulic_potential = variables['hydraulic_potential']
		self.latent_heat = variables['latent_heat']
		self.source_term = variables['source_term']
		self.water_density = variables['water_density']
		self.minimum_drainage_thickness = variables['minimum_drainage_thickness']
		self.closure_coefficient = variables['closure_coefficient']
		
		#Conduit
		self.S = variables['S']
		self.channel_constant = variables['channel_constant']

		#Sheet
		self.hydraulic_potential_sheet = variables['hydraulic_potential']
		self.ev = variables['ev']
		self.percolation_threshold = variables['percolation_threshold']
		self.geothermal_heat_flux = variables['geothermal_heat_flux']
		self.h0 = variables['h0']
		self.background_conductivity = variables['background_conductivity']
		self.sheet_conductivity = variables['sheet_conductivity']
		self.conduit_spacing = variables['conduit_spacing']

		self.sheet_discharge = 0.0

		self.model = model

		# Cavity sheet should be changed to take input as a dictionary
		self.conduit_system = ConduitSystem(hydraulic_potential = self.hydraulic_potential, latent_heat = self.latent_heat, source_term = 0.0, water_density = self.water_density, minimum_drainage_thickness = self.minimum_drainage_thickness, S = self.S, conduit_spacing = self.conduit_spacing, channel_constant = self.channel_constant, closure_coefficient = self.closure_coefficient, model = model)
		self.sheet_system = CavitySheet(water_density = self.water_density, hydraulic_potential = self.hydraulic_potential, background_conductivity = self.background_conductivity, source_term = self.source_term, sheet_conductivity = self.sheet_conductivity, percolation_threshold = self.percolation_threshold, h0 = self.h0, ev = self.ev,minimum_drainage_thickness = self.minimum_drainage_thickness, model = model)

	def update_water_pressure(self):

		model = self.model

		self.sheet_system.update_water_pressure()
		self.water_pressure = self.sheet_system.water_pressure
		self.hydraulic_potential = self.sheet_system.hydraulic_potential
		self.conduit_system.update_water_pressure()
		#self.hydraulic_potential = self.water_pressure + self.water_density*model.g*model.b

	def step(self):

		model = self.model

		def sheet_system_source_term(time):
			return self.source_term(time) - self.conduit_system.exchange_source_term
		self.sheet_system.source_term = sheet_system_source_term
		#self.sheet_system.source_term = self.source_term(model) - self.conduit_system.exchange_source_term

		self.sheet_system.step()
		self.sheet_discharge = self.sheet_system.getDischarge()

		self.update_water_pressure()
		self.hydraulic_potential = self.water_pressure + self.water_density*model.g*model.b

		self.conduit_system.step()
		self.S = self.conduit_system.S

	def getDictionary(self, init = False):

		model = self.model

		if init:
			return {'conduit_spacing':self.conduit_spacing, 'ev':self.ev, 'h0':self.h0,
			'water_density':self.water_density, 'latent_heat':self.latent_heat, 'minimum_drainage_thickness':self.minimum_drainage_thickness,
			'percolation_threshold':self.percolation_threshold, 'background_conductivity':self.background_conductivity, 'geothermal_heat_flux':self.geothermal_heat_flux,
			'sheet_conductivity':self.sheet_conductivity, 'channel_constant':self.channel_constant, 'closure_coefficient':self.closure_coefficient}
		else:
			return {'hydraulic_potential':self.hydraulic_potential.tolist(), 'water_pressure':self.water_pressure.tolist(), 'S':self.S.tolist(),
			'ExchangeTerm':self.conduit_system.exchange_source_term.tolist(), 'SourceTerm':self.source_term(model)}


class ConduitSystem():
	def __init__(self, hydraulic_potential = 0.0, latent_heat = 3.35e5, source_term = 0.0, water_density = 1000.0, minimum_drainage_thickness = 0.0,
		S = 1.0, conduit_spacing = 100.0, closure_coefficient = 1.0e8, channel_constant = 0.1, alpha = 4.0/3.0, beta = 3.0/2.0, model = None):
		#
		# The conduit system is solved under the approximation of zero water pressur in the sheet and the conduit. 
		# This allows for the computation of a source term between the sheet and the conduit that causes equal water pressures.
		# Water pressure is assumed to be positive. When the water pressure is zero, the melt opening term is set to zero to account for partially filled conduits
		#

		self.hydraulic_potential = hydraulic_potential
		self.latent_heat = latent_heat
		self.source_term = source_term
		self.water_density = water_density
		self.minimum_drainage_thickness = minimum_drainage_thickness
		self.S = S
		self.closure_coefficient = closure_coefficient
		self.channel_constant = channel_constant
		self.exchange_source_term = 0.0
		self.conduit_spacing = conduit_spacing
		self.alpha = alpha
		self.beta = beta
		self.model = model


	def update_water_pressure(self):
		#self.water_pressure = model.DrainageSystem.water_pressure
		self.water_pressure = self.hydraulic_potential*0
		#self.water_pressure[np.where(model.DrainageSystem.water_pressure<0)] = model.DrainageSystem.water_pressure[np.where(model.DrainageSystem.water_pressure<0)]


	def step(self):
		model = self.model
		eps = 1.0e-20

		water_pressure_sheet = 0.5*(model.DrainageSystem.water_pressure[1:]+model.DrainageSystem.water_pressure[0:-1])
		ind_underpressure = np.where(water_pressure_sheet<=0) #Indices where water pressure in the sheet is negative

		S_staggered = (self.S[1:]+self.S[0:-1])/2.0 #Set up staggered grid

		sheet_hydraulic_conductivity = model.DrainageSystem.sheet_system.getHydraulicConductivity()
		hydraulic_potential_conduit = self.water_density*model.g*model.b + self.water_pressure
		hydraulic_potential_sheet = self.water_density*model.g*model.b+model.DrainageSystem.water_pressure
		sheet_discharge = model.DrainageSystem.sheet_discharge
		hydraulic_potential_gradient_sheet = (hydraulic_potential_sheet[1:]-hydraulic_potential_sheet[0:-1])/model.dx
		hydraulic_potential_gradient_conduit = (hydraulic_potential_conduit[1:]-hydraulic_potential_conduit[0:-1])/model.dx

		# Calculate opening and closure prefactors and constants
		self.update_water_pressure
		normal_stress = model.rho*model.g*model.H - self.water_pressure

		#Modify normal stress where the sheet water pressure is negative to ensure closure of conduits in those regions (closure coefficient is three orders of magnitude smaller when wp is negative):
		normal_stress[np.where(model.DrainageSystem.water_pressure<0)] = normal_stress[np.where(model.DrainageSystem.water_pressure<0)] - model.DrainageSystem.water_pressure[np.where(model.DrainageSystem.water_pressure<0)]*1.0e3

		opening_term_constant = np.abs( 1.0/(model.rho*self.latent_heat) * np.abs(self.conduit_spacing*( (sheet_discharge[1:] + sheet_discharge[0:-1])/2.0 )) * hydraulic_potential_gradient_sheet )
		opening_term_prefactor = np.abs( 1.0/(model.rho*self.latent_heat) * (np.abs(( self.channel_constant*(S_staggered)**(self.alpha-1)*np.abs(hydraulic_potential_gradient_conduit)**(self.beta-1) ) * hydraulic_potential_gradient_conduit)) )
		closure_term_prefactor = np.sign((normal_stress[1:]+normal_stress[0:-1])/2)*2*(np.abs((normal_stress[1:]+normal_stress[0:-1])/2)/(model.n*self.closure_coefficient))**model.n

		#Semi-implicit time-integration:
		S_staggered_previous_step = S_staggered
		B = opening_term_constant
		A = opening_term_prefactor - closure_term_prefactor
		
		S_staggered = (B*model.dt + S_staggered_previous_step)/(1.0 - A*model.dt)
		
		# Calculate dS_dt (needed for source term)
		opening_term = opening_term_constant + opening_term_prefactor*(S_staggered+S_staggered_previous_step)/2.0
		closure_term = closure_term_prefactor*(S_staggered+S_staggered_previous_step)/2.0
		dS_dt = opening_term - closure_term 

		# Calculate source term
		conduit_discharge = -self.channel_constant*((S_staggered+S_staggered_previous_step)/2.0)**self.alpha*np.abs(hydraulic_potential_gradient_conduit)**(self.beta-1.0)*np.sign(hydraulic_potential_gradient_conduit)
		exchange_source_term_times_conduit_spacing = (dS_dt + np.gradient(conduit_discharge)/model.dx - opening_term*model.rho/self.water_density)
		self.exchange_source_term = exchange_source_term_times_conduit_spacing/self.conduit_spacing

		# Make absolutely sure the channel size is never negative (small negative values could cause unstable behavior).
		S_staggered[np.where(S_staggered<0.0)]=0.0 

		# Back to regular grid:
		S_staggered = np.hstack([S_staggered[0],S_staggered,S_staggered[-1]])
		self.S = (S_staggered[1:]+S_staggered[0:-1])/2
		self.exchange_source_term = np.hstack([0,self.exchange_source_term,0])
		self.exchange_source_term = (self.exchange_source_term[1:]+self.exchange_source_term[0:-1])/2

class HardBed_RSF():
	#
	# Solves hard bed rate-and-state based on Thøgersen et. al (Nature communications 2019). The state parameter is integrated and changes over a length scale dc and/or a time-scale tc.
	# There is an additional term t_closure_zero_thickness to account for the possibility of zero ice thickness. The standard value ensures that "cavities" at zero thickness close quickly.
	#

	def __init__(self, variables, model = None):
	
		self.state_parameter = variables['state_parameter']
		self.C = variables['C']
		self.As = variables['As']
		self.m = variables['m']
		self.q = variables['q']
		self.tc = variables['tc']
		self.dc = variables['dc']
		self.t_closure_zero_thickness = variables['t_closure_zero_thickness']
		self.model = model

	def step(self):
		
		model = self.model
		eps = 1.0e-20
		#sigma_N = model.rho*model.g*model.H - model.DrainageSystem.water_pressure*(model.DrainageSystem.water_pressure>0.0)
		sigma_N = model.rho*model.g*model.H - model.DrainageSystem.water_pressure #Allow for negative water pressure
		sigma_N[np.where(sigma_N<eps)]=eps # We do not solve for uplift when the effective normal stress is negative
		xi = np.abs(model.sliding_velocity)/((self.C**self.m)*(sigma_N**self.m)*self.As)
		alpha = (self.q-1.0)**(self.q-1.0)/(self.q**self.q)
		theta_dagger = (1.0/(1.0 + alpha*xi**self.q))**(1.0/self.m)

		self.state_parameter_derivative = (np.abs(model.sliding_velocity)/self.dc)*(theta_dagger-self.state_parameter) + (theta_dagger-self.state_parameter)/self.tc		

		self.state_parameter_derivative[np.where(model.H<=0)] = self.state_parameter_derivative[np.where(model.H<=0)] + (1-self.state_parameter[np.where(model.H<=0)])/self.t_closure_zero_thickness# Quickly close cavities when ice is gone:
		self.state_parameter_derivative[np.where(self.state_parameter+self.state_parameter_derivative*model.dt < 0.0)] = -self.state_parameter[np.where(self.state_parameter+self.state_parameter_derivative*model.dt < 0.0)]/model.dt #avoid unphysical result by chaning derivative that will pass 0 (although time-step should be small enough so that this is not needed)
		self.state_parameter = self.state_parameter + self.state_parameter_derivative*model.dt # Forward Euler step

		self.state_parameter[np.where(self.state_parameter>1.0)]=1.0

		self.update_friction_coefficient()
		self.update_lateral_drag()


	def update_friction_coefficient(self):
		model = self.model
		eps = 1.0e-20
		velocity = np.abs(model.sliding_velocity)+eps
		self.friction_coefficient = self.state_parameter*(velocity/self.As)**(1.0/self.m-1.0)/self.As # basal friction:


	def update_lateral_drag(self,eps=1.0e-20):
		model = self.model
		velocity = np.abs(model.sliding_velocity)+eps
		self.lateral_drag = 2.0*model.H/model.width * (5.0/(model.A*model.width))**(1.0/3.0)*velocity**(-2.0/3.0) # drag from lateral boundaries

	def getDictionary(self, init = False):
		model = self.model
		if init:
			return {'C':self.C, 'As':self.As, 'm':self.m, 'q':self.q, 'tc':self.tc, 'dc':self.dc, 't_closure_zero_thickness':self.t_closure_zero_thickness}
		else:
			return {'state_parameter':model.FrictionLaw.state_parameter.tolist()}


class Output():
	def __init__(self, output_interval = 1, foldername = 'results', file_format = 'mat', model = None):
		self.ind = 0
		self.output_interval = output_interval
		self.file_format = file_format
		self.foldername = foldername
		self.filename = sys.argv[0]
		self.model = model

	def save(self):

		model = self.model
	
		if self.ind==0:	
			# Set up folder structure and copy run script to src
			if os.path.exists(self.foldername):
				raise Exception('folder (' + self.foldername + ') already exists, aborting to avoid possible overwrite')

			os.mkdir(self.foldername)
			os.mkdir(self.foldername+'/src')
			os.mkdir(self.foldername+'/data')
			os.system('cp ' + self.filename + ' ' + self.foldername + '/src/' + 'run_script.py')				

			# Create dictionary with initial conditions as defined in the different classes
			varDictionary_INIT = model.getDictionary(init = True)
			if(model.FrictionLaw is not None):
				varDictionary_INIT.update(model.FrictionLaw.getDictionary(init = True))
			if(model.DrainageSystem is not None):
				varDictionary_INIT.update(model.DrainageSystem.getDictionary(init = True))

			if self.file_format == 'mat':
				sio.savemat(self.foldername+'/src/INIT.mat', varDictionary_INIT)
			elif self.file_format == 'json':
				with open (self.foldername+'/src/INIT.json', 'w') as file:
					json.dump(varDictionary_INIT,file)

		if(self.ind%self.output_interval==0):
			varDictionary = model.getDictionary(init = False)
			if(model.FrictionLaw is not None):
				varDictionary.update(model.FrictionLaw.getDictionary(init = False))
			if(model.DrainageSystem is not None):
				varDictionary.update(model.DrainageSystem.getDictionary(init = False))

			if self.file_format == 'mat':
				sio.savemat(self.foldername+'/data/'+str(self.ind)+'.mat', varDictionary)
			elif self.file_format == 'json':
				with open (self.foldername+'/data/'+str(self.ind)+'.json', 'w') as file:
					json.dump(varDictionary,file)

		self.ind = self.ind+1


	def load(self, foldername, timestep, file_format):
		# Loads state of model from file. Important note: This only loads the variables, all the initial parameters have to be set.

		if file_format=='json':
			with open(foldername+'/data/' + str(timestep) + '.json') as file:
				data = json.load(file)

		elif file_format=='mat':
			data = sio.loadmat(foldername+'/data/' + str(timestep) + '.mat')

		self.model.t = data.get('t')
		self.model.H = data.get('H')
		self.model.S = data.get('S')
		self.FrictionLaw.state_parameter = data.get('state_parameter')
		self.DrainageSystem.hydraulic_potential = data.get('hydraulic_potential')

class PostProcess():
	# Class for loading of entire datasets and basic plotting of standard figures with matplotlib.

	def __init__(self, foldername, file_format, resolution = 1):

		self.S = np.array([])
		self.H = np.array([])
		self.t = np.array([])
		self.U = np.array([])
		self.state_parameter = np.array([])
		self.hydraulic_potential = np.array([])
		self.file_format = file_format
		self.foldername = foldername
		self.dataLoaded = False
		self.loadedIndices = []
		self.resolution = resolution

		plt.rcParams.update({"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Helvetica"]}) #Which font to choose to make this ready for pip install

		self.loadData()

	def getFileIndices(self):
		allFiles = os.listdir(self.foldername + '/data')
		if self.file_format=='json':
			allIndices = ([int(s.strip('.json')) for s in allFiles])
		elif self.file_format=='mat':
			allIndices = ([int(s.strip('.mat')) for s in allFiles])
		allIndices.sort()
		return allIndices

	def loadDataFile(self,file):
		if self.file_format=='json':
			with open(self.foldername + '/data/' + file) as f:
				data = json.load(f)
		elif self.file_format=='mat':
			data = sio.loadmat(self.foldername + '/data/' + file)
		return data

	def loadData(self):

		if not os.path.exists(self.foldername+'/PostProcess'):
			os.system('mkdir ' +self.foldername+'/PostProcess')

		if self.dataLoaded == False:

			# Load constants
			if self.file_format=='json':
				with open(self.foldername + '/src/INIT.json') as f:
					init = json.load(f)
			elif self.file_format=='mat':
				init = sio.loadmat(self.foldername + '/src/INIT.mat')
			self.x = init['x']
			self.b = init['b']
			self.rho = init['rho']
			self.water_density = init['water_density']
			self.g = init['g']

			# Load time-series
			allIndices = self.getFileIndices()
			allIndices = allIndices[0:-1:self.resolution] # apply given resolution
			loadedFiles = [str(s)+'.'+self.file_format for s in allIndices]

			self.U = np.zeros([np.size(allIndices),np.size(self.x)])
			self.H = np.zeros([np.size(allIndices),np.size(self.x)])
			self.S = np.zeros([np.size(allIndices),np.size(self.x)])
			self.state_parameter = np.zeros([np.size(allIndices),np.size(self.x)])
			self.hydraulic_potential = np.zeros([np.size(allIndices),np.size(self.x)])
			i = 0
			for file in loadedFiles:
				data = self.loadDataFile(file)
				self.U[i,0:] = np.array(data['U']*secondyears)
				self.H[i,0:] = np.array(data['H'])
				self.S[i,0:] = np.array(data['S'])
				self.state_parameter[i,0:] = np.array(data['state_parameter'])
				self.hydraulic_potential[i,0:] = np.array(data['hydraulic_potential']/1e6) #Given in MPa
				self.t = np.append(self.t, data.get('t')/secondyears) #Given in years
				i+=1

			self.loadedIndices = allIndices
			self.dataLoaded = True
			print('Loaded ' + str(np.size(allIndices)) + ' steps in time interval [' + str(self.t[0]) + ',' + str(self.t[-1]) + '] years')

		else: # If called again after initialization (typicallue due to analyzing running simulation), append the new data at the end
			allIndices = self.getFileIndices()
			allIndices.sort()
			allIndices[0:-1:self.resolution] # apply given resolution

			newIndices = list(set(allIndices) - set(self.loadedIndices))
			newIndices.sort()
			newFiles = [str(s)+'.'+self.file_format for s in newIndices]
			t0 = self.t[-1]

			for file in newFiles: # Append new data to the arrays
				data = self.loadDataFile(file)
				self.U = np.append(self.U, data['U']*secondyears, axis=0)
				self.H = np.append(self.H, data['H'], axis=0)
				self.S = np.append(self.S, data['S'], axis=0)
				self.state_parameter = np.append(self.state_parameter, data['state_parameter'], axis=0)
				self.hydraulic_potential = np.append(self.hydraulic_potential, data['hydraulic_potential']/1e6, axis=0)
				self.t = np.append(self.t, data.get('t')/secondyears) #Given in years
			
			self.loadedIndices = newIndices + self.loadedIndices
			print('Loaded ' + str(np.size(newIndices)) + ' steps in time interval [' + str(t0) + ',' + str(self.t[-1]) + '] years')


	def plotAverage(self, variable, printToFile = False):
		# Plots average of given variable where the ice thickness is nonzero
		
		plt.figure()
		var = self.getVariable(variable)
		var[np.where(self.H<=0)]=float('NaN')
		plt.plot(self.t,np.nanmean(var,1))
		plt.xlabel('t [years]')
		plt.ylabel('$\langle$ '+ self.getTitle(variable) + '$\\rangle$' + '[' + self.getUnits(variable) + ']')
		
		if printToFile:
			plt.savefig(self.foldername+'/PostProcess/average_'+ variable +'.eps', format='eps')

	def getVariable(self,variable,printToFile = False):
		# returns variable from string for further processing
		
		if variable == 'U':
			returnVal = self.U
		elif variable == 'S':
			returnVal = self.S
		elif variable == 'H':
			returnVal = self.H
		elif variable == 'state_parameter':
			returnVal = self.state_parameter
		elif variable == 'hydraulic_potential':
			returnVal = self.hydraulic_potential
		elif variable == 'water_pressure':
			returnVal = self.hydraulic_potential - self.water_density*self.g*self.b/1e6
		else:
			raise Exception('variable '+variable+' does not exist')

		return returnVal.copy()


	def getTitle(self,variable):
		# returns title string from string for further processing
		
		if variable == 'U':
			returnVal = 'U'
		elif variable == 'S':
			returnVal = 'S'
		elif variable == 'H':
			returnVal = 'H'
		elif variable == 'state_parameter':
			returnVal = '$\\theta$'
		elif variable == 'hydraulic_potential':
			returnVal = '$\phi$'
		elif variable == 'water_pressure':
			returnVal = '$p_w$'
		else:
			raise Exception('variable '+variable+' does not exist')
		return returnVal

	def getUnits(self,variable):
		# returns variable units from string for further processing
		
		if variable == 'U':
			returnVal = 'm/yr'
		elif variable == 'S':
			returnVal = 'm$^2$'
		elif variable == 'H':
			returnVal = 'm'
		elif variable == 'state_parameter':
			returnVal = ''
		elif variable == 'hydraulic_potential':
			returnVal = 'MPa'
		elif variable == 'water_pressure':
			returnVal = 'MPa'
		else:
			raise Exception('variable '+variable+' does not exist')
		return returnVal


	def plotPcolor(self,variable,printToFile = False):
		# Plots spatiotemporal pcolor of given variable where the ice thickness is nonzero
		plt.figure()
		var = self.getVariable(variable)
		var[np.where(self.H<=0)]=float('NaN')
		plt.pcolor(self.x,self.t,var)
		col = plt.colorbar()
		col.set_label(self.getTitle(variable) + '[' + self.getUnits(variable) + ']')
		plt.xlabel('x [m]')
		plt.ylabel('t [years]')

		if printToFile:
			plt.savefig(self.foldername+'/PostProcess/pcolor_'+ variable +'.png', format='png', dpi=1500)


	def show(self):
		plt.show()

