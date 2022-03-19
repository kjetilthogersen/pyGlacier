import numpy as np

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

		#model = self.model
		eps = 1.0e-20
		#sigma_N = model.rho*model.g*model.H - model.DrainageSystem.water_pressure*(model.DrainageSystem.water_pressure>0.0)
		sigma_N = self.model.rho*self.model.g*self.model.H - self.model.DrainageSystem.water_pressure #Allow for negative water pressure
		sigma_N[np.where(sigma_N<eps)]=eps # We do not solve for uplift when the effective normal stress is negative
		xi = np.abs(self.model.sliding_velocity)/((self.C**self.m)*(sigma_N**self.m)*self.As)
		alpha = (self.q-1.0)**(self.q-1.0)/(self.q**self.q)
		theta_dagger = (1.0/(1.0 + alpha*xi**self.q))**(1.0/self.m)

		self.state_parameter_derivative = (np.abs(self.model.sliding_velocity)/self.dc)*(theta_dagger-self.state_parameter) + (theta_dagger-self.state_parameter)/self.tc

		self.state_parameter_derivative[np.where(self.model.H<=0)] = self.state_parameter_derivative[np.where(self.model.H<=0)] + (1-self.state_parameter[np.where(self.model.H<=0)])/self.t_closure_zero_thickness# Quickly close cavities when ice is gone:
		self.state_parameter_derivative[np.where(self.state_parameter+self.state_parameter_derivative*self.model.dt < 0.0)] = -self.state_parameter[np.where(self.state_parameter+self.state_parameter_derivative*self.model.dt < 0.0)]/self.model.dt #avoid unphysical result by chaning derivative that will pass 0 (although time-step should be small enough so that this is not needed)
		self.state_parameter = self.state_parameter + self.state_parameter_derivative*self.model.dt # Forward Euler step

		self.state_parameter[np.where(self.state_parameter>1.0)]=1.0

		self.update_friction_coefficient()
		self.update_lateral_drag()


	def update_friction_coefficient(self, eps=1.0e-20):
		velocity = np.abs(self.model.sliding_velocity)+eps
		self.friction_coefficient = self.state_parameter*(velocity/self.As)**(1.0/self.m-1.0)/self.As # basal friction:


	def update_lateral_drag(self, eps=1.0e-20):
		velocity = np.abs(self.model.sliding_velocity)+eps
		self.lateral_drag = 2.0*self.model.H/self.model.width * (5.0/(self.model.A*self.model.width))**(1.0/3.0)*velocity**(-2.0/3.0) # drag from lateral boundaries

	def getDictionary(self, init = False):
		if init:
			return {'C':self.C, 'As':self.As, 'm':self.m, 'q':self.q, 'tc':self.tc, 'dc':self.dc, 't_closure_zero_thickness':self.t_closure_zero_thickness}
		else:
			return {'state_parameter':self.model.FrictionLaw.state_parameter.tolist()}
