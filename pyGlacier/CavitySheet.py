import numpy as np
from .solvers.oneDimensionalSolvers import step_diffusion_1d

class CavitySheet():
	#
	# Solves for the hydraulic potential in a cavity sheet where the sheet hydraulic conductivity is set by the state parameter from the sliding law.
	#
	def zero_func(model): #Default zero function (if function not supplied by user)
		return model.b*0


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
		dissipation_friction = np.abs(model.sliding_velocity**2.0*model.FrictionLaw.friction_coefficient(model.sliding_velocity))
		melt_rate = 1.0/(self.latent_heat*model.rho)*(self.geothermal_heat_flux + dissipation_friction) # Calculate melt rate
		effective_conductivity = self.getHydraulicConductivity()

		D = effective_conductivity*self.water_density*model.g/self.ev
		D_staggered = (D[1:]+D[0:-1])/2.0

		beta = self.water_density*model.g/self.ev*(self.h0*model.FrictionLaw.state_parameter_derivative + self.source_term(model) + melt_rate + source_term_from_conduit)
		#beta = self.water_density*model.g/self.ev*(1/(model.FrictionLaw.state_parameter+1.0e-9)*self.h0*model.FrictionLaw.state_parameter_derivative + self.source_term(model) + melt_rate + source_term_from_conduit) # with the logarithmic term
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
		effective_conductivity = self.background_conductivity*np.ones(np.size(model.b)) + self.sheet_conductivity*(1.0-model.FrictionLaw.state_parameter)**3.0*perc_fun
		#effective_conductivity = self.background_conductivity*np.ones(np.size(model.b)) + self.sheet_conductivity*(self.h0*(1.0-model.FrictionLaw.state_parameter))**3.0*perc_fun
		#effective_conductivity = self.background_conductivity*np.ones(np.size(model.b)) - self.sheet_conductivity*(self.h0*np.log(model.FrictionLaw.state_parameter))**3.0*perc_fun #log-term for opening
		return effective_conductivity

	def percolation_function(self):
		model = self.model
		return .5*(np.tanh((self.percolation_threshold-model.FrictionLaw.state_parameter)*50.0)+1.0)
		#return .5*(np.tanh((self.percolation_threshold-model.FrictionLaw.state_parameter)*5.0)+1.0)

	def getDictionary(self, init = False):
		model = self.model
		if init:
			return {'ev':self.ev, 'h0':self.h0, 'water_density':self.water_density, 'latent_heat':self.latent_heat, 'minimum_drainage_thickness':self.minimum_drainage_thickness,
			'percolation_threshold':self.percolation_threshold, 'background_conductivity':self.background_conductivity, 'geothermal_heat_flux':self.geothermal_heat_flux,
			'sheet_conductivity':self.sheet_conductivity, 'channel_constant':self.channel_constant, 'closure_coefficient':self.closure_coefficient}
		else:
			return {'hydraulic_potential':self.hydraulic_potential.tolist(), 'water_pressure':self.water_pressure.tolist(), 'SourceTerm':self.source_term(model)}
