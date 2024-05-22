import numpy as np

from .ConduitSystem import ConduitSystem
from .CavitySheet import CavitySheet

class CoupledConduitCavitySheet():
	def __init__(self, variables, model):
		#
		# Sets up a coupled cavity conduit drainage system. Dependency on CavitySheet and ConduitSystem classes
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
		self.conduit_spacing = variables['conduit_spacing']
		self.conduit_sheet_contribution = variables['conduit_sheet_contribution']
		self.ConduitPressure =  variables['ConduitPressure']

		#Sheet
		self.hydraulic_potential_sheet = variables['hydraulic_potential']
		self.ev = variables['ev']
		self.percolation_threshold = variables['percolation_threshold']
		self.geothermal_heat_flux = variables['geothermal_heat_flux']
		self.h0 = variables['h0']
		self.background_conductivity = variables['background_conductivity']
		self.sheet_conductivity = variables['sheet_conductivity']

		self.sheet_discharge = 0.0

		self.model = model

		# Cavity sheet should be changed to take input as a dictionary
		self.conduit_system = ConduitSystem(hydraulic_potential = self.hydraulic_potential, latent_heat = self.latent_heat, source_term = 0.0, water_density = self.water_density, minimum_drainage_thickness = self.minimum_drainage_thickness, S = self.S, conduit_spacing = self.conduit_spacing, conduit_sheet_contribution = self.conduit_sheet_contribution, channel_constant = self.channel_constant, closure_coefficient = self.closure_coefficient, ConduitPressure = self.ConduitPressure, model = model)
		self.sheet_system = CavitySheet(water_density = self.water_density, hydraulic_potential = self.hydraulic_potential, background_conductivity = self.background_conductivity, source_term = self.source_term, sheet_conductivity = self.sheet_conductivity, percolation_threshold = self.percolation_threshold, h0 = self.h0, ev = self.ev,minimum_drainage_thickness = self.minimum_drainage_thickness, model = model)

	def update_water_pressure(self):

		self.sheet_system.update_water_pressure()
		self.water_pressure = self.sheet_system.water_pressure
		self.hydraulic_potential = self.sheet_system.hydraulic_potential
		self.conduit_system.update_water_pressure()
		#self.hydraulic_potential = self.water_pressure + self.water_density*model.g*model.b

	def step(self):

		def sheet_system_source_term(time):
			return self.source_term(time) - self.conduit_system.exchange_source_term
		self.sheet_system.source_term = sheet_system_source_term
		#self.sheet_system.source_term = self.source_term(model) - self.conduit_system.exchange_source_term

		self.sheet_system.step()
		self.sheet_discharge = self.sheet_system.getDischarge()

		self.update_water_pressure()
		self.hydraulic_potential = self.water_pressure + self.water_density*self.model.g*self.model.b

		self.conduit_system.step()
		self.S = self.conduit_system.S

	def getDictionary(self, init = False):

		if init:
			return {'conduit_spacing':self.conduit_spacing, 'conduit_sheet_contribution':self.conduit_sheet_contribution, 'ev':self.ev, 'h0':self.h0,
			'water_density':self.water_density, 'latent_heat':self.latent_heat, 'minimum_drainage_thickness':self.minimum_drainage_thickness,
			'percolation_threshold':self.percolation_threshold, 'background_conductivity':self.background_conductivity, 'geothermal_heat_flux':self.geothermal_heat_flux,
			'sheet_conductivity':self.sheet_conductivity, 'channel_constant':self.channel_constant, 'closure_coefficient':self.closure_coefficient}
		else:
			return {'hydraulic_potential':self.hydraulic_potential.tolist(), 'water_pressure':self.water_pressure.tolist(), 'S':self.S.tolist(),
			'ExchangeTerm':self.conduit_system.exchange_source_term.tolist(), 'SourceTerm':self.source_term(self.model)}
