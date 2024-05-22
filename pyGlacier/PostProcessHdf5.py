import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.font_manager
import math
from scipy.signal import find_peaks
import tables as tb
secondyears = 60*60*24*365.25

#TODO: change the PostProcess class to use hdf5 file format as well. write functions to load all data into memory, but also to return specific data.

class PostProcess():
	# Class for loading of entire datasets and basic plotting of standard figures with matplotlib.

	def __init__(self, foldername, file_format, resolution = 1):

		self.S = np.array([])
		self.H = np.array([])
		self.t = np.array([])
		self.U = np.array([])
		self.U_SSA = np.array([])
		self.state_parameter = np.array([])
		self.hydraulic_potential = np.array([])
		self.file_format = file_format
		self.foldername = foldername
		self.dataLoaded = False
		self.loadedIndices = []
		self.resolution = resolution

		#plt.rcParams.update({"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Helvetica"]}) #Which font to choose to make this ready for pip install
		#plt.rcParams.update({"text.usetex": True}) #Which font to choose to make this ready for pip install

		self.loadData()

	def loadData(self):

		if not os.path.exists(self.foldername+'/PostProcess'):
			os.system('mkdir ' +self.foldername+'/PostProcess')

		# Load constants
			
		data = tb.open_file(self.foldername + '/data/out.h5')
		
		self.t = [x for x in data.root.full.t]
		self.t = np.array(self.t)/secondyears
		self.x = [x for x in data.root.INIT.x]
		self.x = np.array(self.x)
		self.b = [x for x in data.root.INIT.b]
		self.b = np.array(self.b)
		self.rho = np.array([x for x in data.root.INIT.rho])
		self.water_density = np.array([x for x in data.root.INIT.water_density])
		self.g = np.array([x for x in data.root.INIT.g])
		self.sheet_conductivity = np.array([x for x in data.root.INIT.sheet_conductivity])
		self.percolation_threshold = np.array([x for x in data.root.INIT.percolation_threshold])
		self.width = np.array([x for x in data.root.INIT.width])
		self.A = np.array([x for x in data.root.INIT.A])
		self.As = np.array([x for x in data.root.INIT.As])
		self.m = np.array([x for x in data.root.INIT.m])
		self.n = np.array([x for x in data.root.INIT.n])
		self.C = np.array([x for x in data.root.INIT.C])
		self.channel_constant = np.array([x for x in data.root.INIT.channel_constant])
		self.background_conductivity = np.array([x for x in data.root.INIT.background_conductivity])
			

		# Load time-series

		self.U = [x for x in data.root.full.U]
		self.U = np.array(self.U)*secondyears
		self.Sliding = [x for x in data.root.full.sliding_velocity]
		self.Sliding = np.array(self.Sliding)*secondyears
		self.U_SSA = [x for x in data.root.full.U_SSA]
		self.U_SSA = np.array(self.U_SSA)*secondyears
		self.H = [x for x in data.root.full.H]
		self.H = np.array(self.H)
		self.S = [x for x in data.root.full.S]
		self.S = np.array(self.S)
		self.state_parameter = [x for x in data.root.full.state_parameter]
		self.state_parameter = np.array(self.state_parameter)
		self.hydraulic_potential =  [x for x in data.root.full.hydraulic_potential]
		self.hydraulic_potential = np.array(self.hydraulic_potential)/1e6
		self.ExchangeTerm = [x for x in data.root.full.ExchangeTerm]
		self.ExchangeTerm = np.array(self.ExchangeTerm)*secondyears
		self.Tau_b = np.array((1.0-self.state_parameter)*(np.abs(self.U_SSA)/secondyears/self.As)**(1.0/self.m)/1e6)
		

	def findAverageMaxima(self,variable,height = 100.0):
		var = self.getAverage(variable)
		t_peaks,_ = find_peaks(var,height=height)
		return self.t[t_peaks],var[t_peaks]

	def findMaxMaxima(self,variable,height = 100.0):
		var = self.getMax(variable)
		t_peaks,_ = find_peaks(var,height=height)
		return self.t[t_peaks],var[t_peaks]

	def findMaxAbove(self,variable,height = 100.0):
		var = self.getMax(variable)
		var[np.where(var>=height)]=height
		t_peaks,_ = find_peaks(var,height=height)
		return self.t[t_peaks],var[t_peaks]

	def findAverageAbove(self,variable,height = 100.0):
		var = self.getAverage(variable)
		var[np.where(var>=height)]=height
		t_peaks,_ = find_peaks(var,height=height)
		return self.t[t_peaks],var[t_peaks]

	def getAverage(self,variable):
		var = self.getVariable(variable)
		var[np.where(self.H<=0)]=float('NaN')
		return np.nanmean(var,1)

	def getMax(self,variable):
		var = self.getVariable(variable)
		var[np.where(self.H<=0)]=float('NaN')
		return np.nanmax(var,1)

	def plotAverage(self, variable, printToFile = False, logscale = False):
		# Plots average of given variable where the ice thickness is nonzero
		
		#plt.figure()
		var = self.getVariable(variable)
		var[np.where(self.H<=0)]=float('NaN')
		plt.plot(self.t,np.nanmean(var,1))
		plt.xlabel('t [years]')
		plt.ylabel('$\langle$ '+ self.getTitle(variable) + '$\\rangle$' + '[' + self.getUnits(variable) + ']')

		if logscale:
			plt.yscale('log')
		
		if printToFile:
			plt.savefig(self.foldername+'/PostProcess/average_'+ variable +'.eps', format='eps')
			

	def plotMax(self, variable, printToFile = False):
		# Plots max of given variable where the ice thickness is nonzero
		var = self.getMax(variable)
		plt.plot(self.t,np.nanmax(var,1))
		plt.xlabel('t [years]')
		plt.ylabel('$max($ '+ self.getTitle(variable) + '$)$' + '[' + self.getUnits(variable) + ']')
		
		if printToFile:
			plt.savefig(self.foldername+'/PostProcess/max_'+ variable +'.eps', format='eps')


	def getVariable(self,variable):
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
		elif variable == 'effective_normal_stress':
			returnVal = self.rho*self.H*self.g/1e6 - (self.hydraulic_potential - self.water_density*self.g*self.b/1e6)
		elif variable == 'ExchangeTerm':
			returnVal = self.ExchangeTerm
		elif variable == 'Tau_b':
			returnVal = self.Tau_b
		elif variable == 'iken':
			returnVal = self.C*(self.rho*self.H*self.g/1e6 - (self.hydraulic_potential - self.water_density*self.g*self.b/1e6))
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
		elif variable == 'effective_normal_stress':
			returnVal = '$\\sigma_N$'
		elif variable == 'ExchangeTerm':
			returnVal = 'Exchange source term'
		elif variable == 'Tau_b':
			returnVal = 'Basal Shear Stress'
		elif variable == 'iken':
			returnVal = 'C*N'
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
		elif variable == 'effective_normal_stress':
			returnVal = 'MPa'
		elif variable == 'ExchangeTerm':
			returnVal = 'm/yr'
		elif variable == 'Tau_b':
			returnVal = 'MPa'
		elif variable == 'iken':
			returnVal = 'MPa'
		else:
			raise Exception('variable '+variable+' does not exist')
		return returnVal


	def plotPcolor(self,variable,printToFile = False, logscale = False, transpose = False, colorbar = False, xmin=0, xmax=2000):
		# Plots spatiotemporal pcolor of given variable where the ice thickness is nonzero
		#plt.figure()
		var = self.getVariable(variable)
#		var[np.where(self.H<=0)]=float('NaN')
		var = np.ma.masked_array(var,self.H<=0)
		
		var = var[np.transpose(np.where(self.t>xmin)),:]
		var = np.squeeze(var)
		tval = self.t[np.where(self.t>xmin)]
		tval = np.squeeze(tval)
		
		var = var[np.transpose(np.where(tval<xmax)),:]
		var = np.squeeze(var)
		tval = tval[np.where(tval<xmax)]
		tval = np.squeeze(tval)
		
		if transpose:
			xval = tval
			yval = self.x
			var = var.transpose()

		else:
			xval = self.x
			yval = tval
	
		if not logscale:
			pcol = plt.pcolormesh(xval,yval,var)
		else:
			pcol = plt.pcolormesh(xval,yval,var,norm=LogNorm())

		pcol.set_rasterized(True)

		if colorbar:
			col = plt.colorbar()
			col.set_label(self.getTitle(variable) + '[' + self.getUnits(variable) + ']')

		if transpose:
			plt.ylabel('x [m]')
			plt.xlabel('t [years]')
		else:
			plt.xlabel('x [m]')
			plt.ylabel('t [years]')

		if printToFile:
			plt.gca().set_rasterized(True)			
			plt.savefig(self.foldername+'/PostProcess/pcolor_'+ variable +'.png', format='png', dpi=1500)

		return pcol
		
	def show(self):
		plt.show()

	def close(self):
		plt.close()


	def show(self):
		plt.show()

	def close(self):
		plt.close()

