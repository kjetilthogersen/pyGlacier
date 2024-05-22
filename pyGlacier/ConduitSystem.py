import numpy as np

class ConduitSystem():
	def __init__(self, hydraulic_potential = 0.0, latent_heat = 3.35e5, source_term = 0.0, water_density = 1000.0, minimum_drainage_thickness = 0.0,
		S = 1.0, conduit_spacing = 100.0,conduit_sheet_contribution = 2.0, closure_coefficient = 1.0e8, channel_constant = 0.1, alpha = 4.0/3.0, beta = 3.0/2.0, ConduitPressure='Zero', model = None):
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
		self.conduit_sheet_contribution = conduit_sheet_contribution
		self.alpha = alpha
		self.beta = beta
		self.model = model
		self.ConduitPressure = ConduitPressure


	def update_water_pressure(self):
		model = self.model
		
		
		# self.water_pressure = model.DrainageSystem.water_pressure
		if self.ConduitPressure == 'Zero':
			self.water_pressure = self.hydraulic_potential*0
		elif self.ConduitPressure == 'Sheet':
			self.water_pressure = model.DrainageSystem.water_pressure
		
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
		
		opening_term_constant = np.abs( 1.0/(model.rho*self.latent_heat) * np.abs(self.conduit_sheet_contribution*( (sheet_discharge[1:] + sheet_discharge[0:-1])/2.0 )) * hydraulic_potential_gradient_sheet )
		opening_term_prefactor = np.abs( 1.0/(model.rho*self.latent_heat) * (np.abs(( self.channel_constant*(S_staggered)**(self.alpha-1)*np.abs(hydraulic_potential_gradient_conduit)**(self.beta-1.0) ) * hydraulic_potential_gradient_conduit)) )

		closure_term_prefactor = np.sign((normal_stress[1:]+normal_stress[0:-1])/2)*2*(np.abs((normal_stress[1:]+normal_stress[0:-1])/2)/(model.n*self.closure_coefficient))**model.n

		#Semi-implicit time-integration:
		S_staggered_previous_step = S_staggered
		B = opening_term_constant
		A = opening_term_prefactor - closure_term_prefactor

		S_staggered = (B*model.dt + S_staggered_previous_step)/(1.0 - A*model.dt)
		S_staggered[np.where(S_staggered<0.0)]=0.0
		S_staggered[np.where(S_staggered>12.0)]=12.0

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
		S_staggered[np.where(S_staggered>12.0)]=12.0
		# S_staggered[np.where(S_staggered>1.0)]=1.0

		# Back to regular grid:
		S_staggered = np.hstack([S_staggered[0],S_staggered,S_staggered[-1]])
		self.S = (S_staggered[1:]+S_staggered[0:-1])/2
		self.exchange_source_term = np.hstack([0,self.exchange_source_term,0])
		self.exchange_source_term = (self.exchange_source_term[1:]+self.exchange_source_term[0:-1])/2
