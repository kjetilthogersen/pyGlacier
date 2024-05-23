####
import pyGlacier as pg
import matplotlib.pyplot as plt
import numpy as np

# Initialize flowline glacier
x = np.linspace(0,2e4,1000)
dx = x[1]-x[0]
b = 2000.0*np.exp(-x/15000)

def SMB(model):
	return (model.b+model.H-1600.0)*1e-3/(pg.secondyears)*4

def source_term(model):
	return 1.0e-8*(np.sin(model.t*np.pi/pg.secondyears)**2.0)

# Set up variable dictionary:
variables = {
'solver': 
	{'ID': 'coupled',
	'variables':{
		'SMB': SMB,
		'rho': 900.0,
		'g': 9.8,
		'n': 3.0,
		'width': 1.2e3,
		't': 0.0,
		'A': 2.4e-24,
		'H': 0*x,
		'b': b,
		'dx': dx,
		'dt': 1e3},
	'boundaryConditions':{
		'left':{
			'type': 'surface slope',
			'val': 0},
		'right':{
			'type': 'thickness',
			'val': 0} } },
'DrainageSystem': 
	{'ID': 'CoupledConduitCavitySheet',
	'variables': {
		'hydraulic_potential':1000.0*9.8*b + 0*1000.0*1000.0*9.8,
		'latent_heat': 3.0e5,
		'source_term': source_term,
		'water_density': 1000.0,
		'minimum_drainage_thickness': 1.0,
		'S': 0*b,
		'ev': 1.0e-2,
		'percolation_threshold': 0.5,
		'geothermal_heat_flux': 0.0,
		'h0': 0.1,
		'background_conductivity': 1.0e-8,
		'sheet_conductivity': 5.0e-8,
		'channel_constant': 0.01,
		'conduit_spacing': 1000.0,
		'conduit_sheet_contribution': 2.0,
		'ConduitPressure': 'Sheet',
		'closure_coefficient': 5.3e7 } },
'FrictionLaw': 
	{'ID': 'HardBed_RSF',
	'variables':{ 
		'state_parameter': np.zeros(np.size(b)),
		'As': 1.0e-23,
		'm': 3.0,
		'q': 2.5,
		'C': 0.3,
		'tc': 1.0e6,
		'dc': 1.0,
		't_closure_zero_thickness': 1.0e5 } },
'Output': 
	{'ID': 'standard',
	'foldername': 'run_benchmarkSIA-SSA',
	'output_interval': 100,
	'reduced': True,
	'reduced_output_interval': 1,
	'flush_interval': 100,
	'file_format': 'hdf5'} }

# Initialize model:
model = pg.Flowline(variables = variables)
model.runInitializeSIA(dt = 1e6, t_max = 3*3.15e7)

# run with adaptive time-stepping:
model.runAdaptive(t_max = 1/10*3.15e7, dt_min = 10, dt_max = 1e3, error_tolerance = 1e-3, interval = 10)