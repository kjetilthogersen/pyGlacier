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
#import matplotlib as plt
#import time
import scipy.io as sio

def step_diffusion_1d(dt,dx,phi,sourceTerm,D,left_boundary_condition,left_boundary_condition_value,right_boundary_condition,right_boundary_condition_value):
	#
	# Solves 1D diffusion on staggered grid; give the midpotins only, not the artificial points outside the boundary, they are specified by the given boundary condition. 
	# Input values are dt, dx, phi on the nodes, source term on the nodes, diffusion coefficient vector on midpoints (size N+1)
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


def step_elliptical_1d_Old(dx,W,alpha,beta,left_boundary_condition,left_boundary_condition_value,right_boundary_condition,right_boundary_condition_value):
	#
	# Solves 1D elliptical equation on staggered grid (@TODO: give equation)
	# (W(x) u_x)_x - alpha(x) u = beta(x)
	#
	def boundary_modifier(x,val):
		return {
			'Dirichlet': [val,0,val],
			'vonNeuman': [-val,val,-val]
		}[x]
	N_intervals = np.size(alpha)-1
	val = np.mean(np.abs(2*W+dx**2.0+alpha))
	left_modifier = boundary_modifier(left_boundary_condition,val)
	right_modifier = boundary_modifier(right_boundary_condition,val)
	rhs = dx*dx*beta
	rhs[0] = left_modifier[2]*left_boundary_condition_value
	rhs[np.size(rhs)-1] = right_modifier[2]*right_boundary_condition_value
	A = csc_matrix(spdiags( [ np.hstack((W[0:N_intervals-1],right_modifier[1],0)),  np.hstack((left_modifier[0],-(W[0:N_intervals-1]+W[1:N_intervals]+alpha[1:N_intervals]*dx*dx),right_modifier[0])),   np.hstack((0,left_modifier[1],W[1:N_intervals]))  ], [-1,0,1], N_intervals+1, N_intervals+1))
	u = dsolve.spsolve(A, rhs, use_umfpack=True)
	return u

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
	def __init__(self, A, H, b, dx, dt, rho = 900.0, g = 9.8, n = 3.0, width = 2.5e3, solver = 'coupled', SMB = 0.0, DrainageSystem = False, FrictionLaw = False, Output = False, t = 0.0):
		self.H = H
		self.b = b
		self.rho = rho
		self.g = g
		self.n = n
		self.A = A
		self.dx = dx
		self.dt = dt
		self.width = width
		self.solver = solver #SIA, SSA, coupled
		self.SMB = SMB

		self.FrictionLaw = FrictionLaw
		self.DrainageSystem = DrainageSystem
		self.Output = Output

		if(DrainageSystem != False):
			self.DrainageSystem.update_water_pressure(self)

		self.sliding_velocity = np.zeros(np.size(b))

		self.t = t

	def step_massContinuity(self):
		# Calculates the mass continuity for a given diffusion coefficients from the
		# the shallow ice approximation and the shallow shelf approximation as well as a given surface mass balance
		h = self.H+self.b
		D_staggered = (self.D[1:]+self.D[0:-1])/2.0 #Staggered grid
		h = step_advection_diffusion_1d(dt = self.dt, dx = self.dx, phi = h, sourceTerm = self.SMB, D = D_staggered, Vphi = -self.sliding_velocity*self.H, left_boundary_condition = 'vonNeuman',left_boundary_condition_value = 0.0, right_boundary_condition = 'vonNeuman', right_boundary_condition_value = 0.0)
		#h = step_diffusion_1d(dt = self.dt, dx = self.dx, phi = h, sourceTerm = self.SMB, D = D_staggered, left_boundary_condition = 'vonNeuman',left_boundary_condition_value = 0.0, right_boundary_condition = 'vonNeuman', right_boundary_condition_value = 0.0)
		self.H = h-self.b
		self.H[np.where(self.H<=0)] = 0 # Ice thickness cannot be below zero


	def step(self):

		if((self.solver == 'SIA') or (self.solver == 'coupled')):
			self.update_sia_flowline()

		if((self.solver == 'SSA') or (self.solver == 'coupled')):
			self.update_ssa_flowline()
		
		if(self.solver == 'coupled'):
			self.U = self.u_SIA + self.u_SSA
			self.sliding_velocity = self.u_SSA
			self.D = self.D_SIA

		if(self.solver == 'SIA'):
			self.U = self.u_SIA 
			self.sliding_velocity = np.zeros(np.size(self.u_SIA))
			self.D = self.D_SIA
			self.VH = np.zeros(np.size(self.D_SIA))

		if(self.solver == 'SSA'):
			self.U = self.u_SSA
			self.sliding_velocity = self.u_SSA
			self.D = np.zeros(np.size(self.D_SSA))
		
		if((self.solver == 'SSA') or (self.solver == 'coupled')):
			self.FrictionLaw.step(self)
			
		self.DrainageSystem.step(self)
		self.step_massContinuity()

		if(self.Output != False):
			self.Output.save(self)
		
		self.t = self.t + self.dt


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


	def update_ssa_flowline(self,tol=1.0e-3,itermax=100,eps=1.0e-20):
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

			self.FrictionLaw.update_friction_coefficient(self) #Update friction coefficient
			self.FrictionLaw.update_lateral_drag(self) #Update lateral drag

			alpha = self.FrictionLaw.friction_coefficient + self.FrictionLaw.lateral_drag #Combine basal and lateral drag in a single coefficient

			dudx = (u[1:] - u[0:-1])/self.dx # On staggered grid
			dudx_sqr_reg = dudx**2.0 + eps**2.0
			
			W = 2*self.A**(-1.0/self.n)*(self.H[1:] + self.H[0:-1])/2.0*dudx_sqr_reg**(((1.0/self.n)-1.0)/2.0)
			W = np.hstack([W,W[-1]])
			u_new = step_elliptical_1d(dx = self.dx, W = W, alpha = alpha, beta = driving_stress, gamma = driving_stress*0, left_boundary_condition = 'Dirichlet',left_boundary_condition_value = 0.0, right_boundary_condition = 'Dirichlet', right_boundary_condition_value = 0.0)
			rel_error = np.mean(  np.abs(u_new-u)/(np.abs(u_new)+eps) )

			u = u_new
			i = i+1
		
		self.u_SSA = u
		self.D_SSA = self.u_SSA*self.H

class CavitySheet():
	#
	# Solves for the hydraulic potential in a cavity sheet where the sheet hydraulic conductivity is set by the state parameter from the sliding law.
	# 
	def __init__(self,hydraulic_potential = 0.0, ev = 1.0e-3, percolation_threshold = 0.6, geothermal_heat_flux = 0.0, h0 = 1.0, water_viscosity = 1.0e-3, latent_heat = 3.35e5, source_term = 1.0e-9, background_conductivity = 1.0e-15, sheet_conductivity = 1.0e-6, water_density = 1000.0, minimum_drainage_thickness = 1.0):
		
		self.hydraulic_potential = hydraulic_potential
		self.ev = ev
		self.percolation_threshold = percolation_threshold
		self.geothermal_heat_flux = geothermal_heat_flux
		self.h0 = h0
		self.water_viscosity = water_viscosity
		self.latent_heat = latent_heat
		self.source_term = source_term
		self.background_conductivity = background_conductivity
		self.sheet_conductivity = sheet_conductivity
		self.water_density = water_density
		self.minimum_drainage_thickness = minimum_drainage_thickness


	def update_water_pressure(self,model):
		self.water_pressure = self.hydraulic_potential - self.water_density*model.g*model.b

		ind = np.where((self.water_pressure>0.0)&(model.H<=self.minimum_drainage_thickness)) # Cannot support water pressure for zero thickness.
		self.hydraulic_potential[ind] = self.water_density*model.g*model.b[ind]#self.hydraulic_potential[ind] - self.water_pressure[ind]
		self.water_pressure[ind]=0.0 # Water pressure is assumed positive
		
		#self.water_pressure[np.where((model.H<self.minimum_drainage_thickness)&(self.water_pressure>0))]=0.0
		#self.hydraulic_potential = self.water_density*model.g*model.b + self.water_pressure

	def step(self,model,source_term_from_conduit = 0.0):

		percolation_function = .5*(np.tanh((self.percolation_threshold-model.FrictionLaw.state_parameter)*5.0)+1.0)
		dissipation_friction = np.abs(model.sliding_velocity**2.0*model.FrictionLaw.friction_coefficient)
		melt_rate = 1.0/(self.latent_heat*model.rho)*(self.geothermal_heat_flux + dissipation_friction) # Calculate melt rate
		effective_conductivity = self.background_conductivity*np.ones(np.size(model.b))
		ind = np.where(self.water_pressure>0.0)
		effective_conductivity[ind] = effective_conductivity[ind] + self.sheet_conductivity*self.h0*(1.0-model.FrictionLaw.state_parameter[ind])**3.0*percolation_function[ind]

		effective_conductivity[np.where(self.water_pressure<0)] = self.background_conductivity

		D = effective_conductivity/self.water_viscosity*self.water_density*model.g/self.ev
		D_staggered = (D[1:]+D[0:-1])/2.0
		beta = self.water_density*model.g/self.ev*(self.h0*model.FrictionLaw.state_parameter_derivative + self.source_term + melt_rate + source_term_from_conduit)
		self.hydraulic_potential = step_diffusion_1d(dt = model.dt, dx = model.dx, phi = self.hydraulic_potential, sourceTerm = beta, D = D_staggered, left_boundary_condition = 'vonNeuman',left_boundary_condition_value = 0.0, right_boundary_condition = 'vonNeuman', right_boundary_condition_value = 0.0)

		self.update_water_pressure(model)

	def getDischarge(self,model): #Discharge in sheet only. excludes the bedrock (background) conductivity
		percolation_function = .5*(np.tanh((self.percolation_threshold-model.FrictionLaw.state_parameter)*5.0)+1.0)
		effective_conductivity = self.background_conductivity*np.ones(np.size(model.b))
		ind = np.where(self.water_pressure>0.0)
		effective_conductivity[ind] = effective_conductivity[ind] + self.sheet_conductivity*self.h0*(1.0-model.FrictionLaw.state_parameter[ind])**3.0*percolation_function[ind]
		effective_conductivity[np.where(self.hydraulic_potential<self.water_density*model.g*model.b)] = self.background_conductivity

		discharge = - effective_conductivity*np.gradient(self.hydraulic_potential/model.dx)
		return discharge
		
	def getHydraulicConductivity(self,model):
		percolation_function = .5*(np.tanh((self.percolation_threshold-model.FrictionLaw.state_parameter)*5.0)+1.0)
		effective_conductivity = self.background_conductivity*np.ones(np.size(model.b))
		ind = np.where(self.water_pressure>0.0)
		effective_conductivity[ind] = effective_conductivity[ind] + self.sheet_conductivity*self.h0*(1.0-model.FrictionLaw.state_parameter[ind])**3.0*percolation_function[ind]
		effective_conductivity[np.where(self.hydraulic_potential<self.water_density*model.g*model.b)] = self.background_conductivity

		return effective_conductivity

class CoupledConduitCavitySheet():
	def __init__(self, hydraulic_potential = 0.0, water_viscosity = 1.0e-3, latent_heat = 3.34e5, source_term = 1.0e-9, water_density = 1000.0, minimum_drainage_thickness = 0.0, S = 1.0, ev = 1.0e-1, percolation_threshold = 0.4, geothermal_heat_flux = 0.0, h0 = 0.1, background_conductivity = 0.0, sheet_conductivity = 1.0e-3, channel_constant = 0.01, conduit_spacing = 1000.0):
		#
		# Sets up a coupled cavity conduit drainage system
		#

		#General parameters
		self.hydraulic_potential = hydraulic_potential
		self.water_viscosity = water_viscosity
		self.latent_heat = latent_heat
		self.source_term = source_term
		self.water_density = water_density
		self.minimum_drainage_thickness = minimum_drainage_thickness
		
		#Conduit
		self.S = S
		self.channel_constant = channel_constant

		#Sheet
		self.hydraulic_potential_sheet = hydraulic_potential
		self.ev = ev
		self.percolation_threshold = percolation_threshold
		self.geothermal_heat_flux = geothermal_heat_flux
		self.h0 = h0
		self.background_conductivity = background_conductivity
		self.sheet_conductivity = sheet_conductivity
		self.conduit_spacing = conduit_spacing

		self.sheet_discharge = 0.0

		self.conduit_system = ConduitSystem(hydraulic_potential = self.hydraulic_potential, water_viscosity = self.water_viscosity, latent_heat = self.latent_heat, source_term = 0.0, water_density = self.water_density, minimum_drainage_thickness = self.minimum_drainage_thickness, S = self.S, conduit_spacing = self.conduit_spacing, channel_constant = self.channel_constant)
		self.sheet_system = CavitySheet(water_density = self.water_density, hydraulic_potential = self.hydraulic_potential, background_conductivity = self.background_conductivity, source_term = self.source_term, sheet_conductivity = self.sheet_conductivity, percolation_threshold = self.percolation_threshold, h0 = self.h0, ev = self.ev,minimum_drainage_thickness = self.minimum_drainage_thickness)

	def update_water_pressure(self,model):
		self.sheet_system.update_water_pressure(model)
		self.conduit_system.update_water_pressure(model)
		self.water_pressure = self.sheet_system.water_pressure
		self.hydraulic_potential = self.sheet_system.hydraulic_potential
		#self.hydraulic_potential = self.water_pressure + self.water_density*model.g*model.b

	def step(self,model):

		self.sheet_system.source_term = self.source_term - self.conduit_system.exchange_source_term

		self.sheet_system.step(model)
		self.sheet_discharge = self.sheet_system.getDischarge(model)

		self.update_water_pressure(model)
		self.hydraulic_potential = self.water_pressure + self.water_density*model.g*model.b

		self.conduit_system.step(model)
		self.S = self.conduit_system.S

		
class ConduitSystem():
	def __init__(self, hydraulic_potential = 0.0, water_viscosity = 1.0e-3, latent_heat = 3.35e5, source_term = 0.0, water_density = 1000.0, minimum_drainage_thickness = 0.0,
		S = 1.0, conduit_spacing = 100.0, claperyon_slope = 7.5e-8, water_heat_capacity = 4.22e3, closure_coefficient = 1.0e8, channel_constant = 0.01, alpha = 4.0/3.0, beta = 3.0/2.0):
		#
		# The conduit system is solved under the approximation of equal water pressur in the sheet and the conduit. 
		# This allows for the computation of a source term between the sheet and the conduit that causes equal water pressures.
		# Water pressure is assumed to be positive. When the water pressure is zero, the melt opening term is set to zero to account for partially filled conduits
		#

		self.hydraulic_potential = hydraulic_potential
		self.water_viscosity = water_viscosity
		self.latent_heat = latent_heat
		self.source_term = source_term
		self.water_density = water_density
		self.minimum_drainage_thickness = minimum_drainage_thickness
		self.S = S

		self.claperyon_slope = claperyon_slope
		self.water_heat_capacity = water_heat_capacity
		self.closure_coefficient = closure_coefficient
		self.channel_constant = channel_constant
	
		self.exchange_source_term = 0.0
		self.conduit_spacing = conduit_spacing*np.ones(np.size(hydraulic_potential))

		self.alpha = alpha
		self.beta = beta


	def update_water_pressure(self,model):
		self.water_pressure = self.hydraulic_potential - self.water_density*model.g*model.b


	def step(self,model):
		eps = 1.0e-20

		S_staggered = (self.S[1:]+self.S[0:-1])/2.0 #Set up staggered grid
		conduit_spacing_staggered = (self.conduit_spacing[1:]+self.conduit_spacing[0:-1])/2.0

		sheet_hydraulic_conductivity = model.DrainageSystem.sheet_system.getHydraulicConductivity(model)
		water_pressure = model.DrainageSystem.water_pressure
		#water_pressure = water_pressure*(water_pressure>0)
		hydraulic_potential = self.water_density*model.g*model.b+water_pressure
		sheet_discharge = model.DrainageSystem.sheet_discharge*np.ones(np.size(water_pressure))
		hydraulic_potential_gradient = (self.hydraulic_potential[1:]-self.hydraulic_potential[0:-1])/model.dx

		D = self.water_heat_capacity*self.claperyon_slope*self.water_density
		#psi_staggered = np.abs(( self.channel_constant*(S_staggered)**self.alpha*np.abs(hydraulic_potential_gradient)**(self.beta-1) + np.abs(self.conduit_spacing*( (sheet_discharge[1:] + sheet_discharge[0:-1])/2.0 )) ) * hydraulic_potential_gradient)
		tmp = conduit_spacing_staggered*(sheet_discharge[1:] + sheet_discharge[0:-1])/2.0
		tmp[np.where(((S_staggered)<=0.0)&((sheet_discharge[1:] + sheet_discharge[0:-1])/2.0*( water_pressure[1:]-water_pressure[0:-1] )/(model.dx)<=0.0))] = 0.0
		#xi_staggered = -D*( -self.channel_constant*(S_staggered)**self.alpha*np.abs(hydraulic_potential_gradient)**(self.beta-1.0)*np.sign(hydraulic_potential_gradient) -tmp )*( water_pressure[1:]-water_pressure[0:-1] )/(model.dx)

		effective_stress = model.rho*model.g*model.H - water_pressure*(water_pressure>0)# + 1.0e5
		#effective_stress[np.where(effective_stress<=1.0e5)]=1.0e5 #neglect overpressure in channels (set to atmospheric pressure)
		effective_stress[np.where(effective_stress<=0.0)]=0.0 #neglect overpressure in channels (set to atmospheric pressure)

		# Calculate opening and closure prefactors and constants
		opening_term_constant = 1.0/(model.rho*self.latent_heat) * (np.abs(( np.abs(conduit_spacing_staggered*( (sheet_discharge[1:] + sheet_discharge[0:-1])/2.0 )) ) * hydraulic_potential_gradient) + D*( -tmp )*( water_pressure[1:]-water_pressure[0:-1] )/(model.dx) )
		opening_term_prefactor = 1.0/(model.rho*self.latent_heat) * (np.abs(( self.channel_constant*(S_staggered)**(self.alpha-1)*np.abs(hydraulic_potential_gradient)**(self.beta-1) ) * hydraulic_potential_gradient) + D*( -self.channel_constant*(S_staggered)**(self.alpha-1)*np.abs(hydraulic_potential_gradient)**(self.beta-1.0)*np.sign(hydraulic_potential_gradient) )*( water_pressure[1:]-water_pressure[0:-1] )/(model.dx) )
		closure_term_prefactor = 2*(np.abs((effective_stress[1:]+effective_stress[0:-1])/2)/(model.n*self.closure_coefficient))**model.n + (model.sliding_velocity[1:] + model.sliding_velocity[0:-1])/2

		#Opening is zero when water pressure is negative:
		opening_term_constant[np.where( (water_pressure[0:-1]+water_pressure[1:])/2 <=eps)] = 0.0
		opening_term_prefactor[np.where( (water_pressure[0:-1]+water_pressure[1:])/2 <=eps)] = 0.0
	
		#Semi-implicit time-integration:
		B = opening_term_constant
		A = opening_term_prefactor - closure_term_prefactor
		S_staggered = (B*model.dt + (S_staggered))/(1 - A*model.dt)
		
		# Find indices with negativ source term:
		opening_term = opening_term_constant + opening_term_prefactor*S_staggered
		closure_term = closure_term_prefactor*S_staggered
		dS_dt = opening_term - closure_term 
		conduit_discharge = -self.channel_constant*(S_staggered)**self.alpha*np.abs(hydraulic_potential_gradient)**(self.beta-1.0)*np.sign(hydraulic_potential_gradient)


		exchange_source_term_times_conduit_spacing = (dS_dt + np.gradient(conduit_discharge)/model.dx - opening_term*model.rho/self.water_density)
		ind_branching = np.where((exchange_source_term_times_conduit_spacing<0))
		ind_closure = np.where((exchange_source_term_times_conduit_spacing>0) & (0.5*(water_pressure[1:]+water_pressure[0:-1])<=0 ))
		

		# To test:
		# Make the correction directly on the conduit spacing and recalculate.
		#
		#


		# Solve for change in conduit spacing:
		#d_conduit_spacing_dt = np.zeros(np.size(conduit_spacing_staggered))
		#tmp = (exchange_source_term_times_conduit_spacing - dS_dt)*conduit_spacing_staggered/(S_staggered+1.0e-5)
		#d_conduit_spacing_dt[ind_branching] = tmp[ind_branching]
		#d_conduit_spacing_dt[ind_closure] = tmp[ind_closure]

		#tmp = np.gradient(conduit_discharge)/model.dx - opening_term*model.rho/self.water_density
		#d_conduit_spacing_dt[ind_branching] = -(exchange_source_term_times_conduit_spacing[ind_branching] - tmp[ind_branching])/(S_staggered[ind_branching]+1e-5)
		#d_conduit_spacing_dt[ind_closure] = -(exchange_source_term_times_conduit_spacing[ind_closure] - tmp[ind_closure])/(S_staggered[ind_closure]+1e-5)
		

		self.exchange_source_term = exchange_source_term_times_conduit_spacing/conduit_spacing_staggered
		self.exchange_source_term[ind_branching] = 0
		self.exchange_source_term[ind_closure] = 0

		#conduit_spacing_staggered = conduit_spacing_staggered + d_conduit_spacing_dt*model.dt
		#conduit_spacing_staggered[S_staggered<1e-5]=1000.0

		conduit_spacing_staggered = np.hstack([conduit_spacing_staggered[0],conduit_spacing_staggered,conduit_spacing_staggered[-1]])
		self.conduit_spacing = (conduit_spacing_staggered[1:]+conduit_spacing_staggered[0:-1])/2

		if False:
			self.exchange_source_term = 1.0/conduit_spacing_staggered*(dS_dt + np.gradient(conduit_discharge)/model.dx - opening_term*model.rho/self.water_density)
			ind = np.where((self.exchange_source_term<0))#&((model.H[1:]+model.H[0:-1])/2>self.minimum_drainage_thickness))
			self.exchange_source_term[ind]=0.0

			ind = np.where((self.exchange_source_term>0) & (0.5*(water_pressure[1:]+water_pressure[0:-1])<0))
			self.exchange_source_term[ind]=0.0 #TODO: fix this correction		

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

	def __init__(self,state_parameter, C = 1.0, As = 1.0e-24, m =3.0, q = 1.7, tc = 5.0e6, dc = 1.0, t_closure_zero_thickness = 1.0e5):
	
		self.C = C
		self.As = As
		self.m = m
		self.q = q
		self.tc = tc
		self.dc = dc
		self.t_closure_zero_thickness = t_closure_zero_thickness
		self.state_parameter = state_parameter


	def step(self,model):
		
		eps = 1.0e-20
		sigma_N = model.rho*model.g*model.H - model.DrainageSystem.water_pressure*(model.DrainageSystem.water_pressure>0.0)
		sigma_N[np.where(sigma_N<eps)]=eps # We do not solve for uplift when the effective normal stress is negative
		xi = np.abs(model.sliding_velocity)/((self.C**self.m)*(sigma_N**self.m)*self.As)
		alpha = (self.q-1.0)**(self.q-1.0)/(self.q**self.q)
		theta_dagger = (1.0/(1.0 + alpha*xi**self.q))**(1.0/self.m)

		self.state_parameter_derivative = (np.abs(model.sliding_velocity)/self.dc)*(theta_dagger-self.state_parameter) + (theta_dagger-self.state_parameter)/self.tc		

		self.state_parameter_derivative[np.where(model.H<=0)] = self.state_parameter_derivative[np.where(model.H<=0)] + (1-self.state_parameter[np.where(model.H<=0)])/self.t_closure_zero_thickness# Quickly close cavities when ice is gone:
		self.state_parameter_derivative[np.where(self.state_parameter+self.state_parameter_derivative*model.dt < 0.0)] = -self.state_parameter[np.where(self.state_parameter+self.state_parameter_derivative*model.dt < 0.0)]/model.dt #avoid unphysical result by chaning derivative that will pass 0 (although time-step should be small enough so that this is not needed)
		self.state_parameter = self.state_parameter + self.state_parameter_derivative*model.dt # Forward Euler step

		self.state_parameter[np.where(self.state_parameter>1.0)]=1.0

		self.update_friction_coefficient(model)
		self.update_lateral_drag(model)


	def update_friction_coefficient(self,model):
		eps = 1.0e-20
		velocity = np.abs(model.sliding_velocity)+eps
		self.friction_coefficient = self.state_parameter*(velocity/self.As)**(1.0/self.m-1.0)/self.As # basal friction:


	def update_lateral_drag(self,model,eps=1.0e-20):
		velocity = np.abs(model.sliding_velocity)+eps
		self.lateral_drag = 2.0*model.H/model.width * (5.0/(model.A*model.width))**(1.0/3.0)*velocity**(-2.0/3.0) # drag from lateral boundaries


class Output():
	def __init__(self,filename, run_script = False, output_interval = 1):
		self.filename = filename
		self.ind = 0
		self.output_interval = output_interval
		if run_script != False:
			with open(run_script, 'r') as myfile:
				self.run_script = myfile.read()
		else:
			self.run_script = ''

	def save(self,model):

		if self.ind==0:
			sio.savemat(self.filename+'_0000_INIT_.mat', {'b':model.b, 'run_script':self.run_script})

		if(self.ind%self.output_interval==0): # Plot
			sio.savemat(self.filename+'_'+str(self.ind)+'.mat', {'U':model.U, 'H':model.H,
				'hydraulic_potential':model.DrainageSystem.hydraulic_potential,
				'state_parameter':model.FrictionLaw.state_parameter, 't':model.t, 'water_pressure':model.DrainageSystem.water_pressure,
				'S':model.DrainageSystem.S, 'ExchangeTerm':model.DrainageSystem.conduit_system.exchange_source_term,
				'conduit_spacing':model.DrainageSystem.conduit_system.conduit_spacing})
		self.ind = self.ind+1





