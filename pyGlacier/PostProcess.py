import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.font_manager
import math
from scipy.signal import find_peaks
secondyears = 60*60*24*365.25

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
			self.x = init['x'][0]
			self.b = init['b'][0]
			self.rho = init['rho']
			self.water_density = init['water_density']
			self.g = init['g']
			self.sheet_conductivity = init['sheet_conductivity']
			self.percolation_threshold = init['percolation_threshold']
			self.width = init['width']
			self.A = init['A']
			self.As = init['As']
			self.m = init['m']
			self.n = init['n']
			self.C = init['C'][0]
			self.channel_constant = init['channel_constant'][0]
			self.background_conductivity = init['background_conductivity'][0]
			self.sheet_conductivity = init['sheet_conductivity'][0]

			# Load time-series
			allIndices = self.getFileIndices()
			allIndices = allIndices[0:-1:self.resolution] # apply given resolution
			loadedFiles = [str(s)+'.'+self.file_format for s in allIndices]

			self.U = np.zeros([np.size(allIndices),np.size(self.x)])
			self.U_SSA = np.zeros([np.size(allIndices),np.size(self.x)])
			self.H = np.zeros([np.size(allIndices),np.size(self.x)])
			self.S = np.zeros([np.size(allIndices),np.size(self.x)])
			self.state_parameter = np.zeros([np.size(allIndices),np.size(self.x)])
			self.hydraulic_potential = np.zeros([np.size(allIndices),np.size(self.x)])
			self.ExchangeTerm = np.zeros([np.size(allIndices),np.size(self.x)])

			i = 0
			for file in loadedFiles:
				data = self.loadDataFile(file)
				self.U[i,0:] = np.array(data['U']*secondyears)
				self.U_SSA[i,0:] = np.array(data['U_SSA']*secondyears)
				self.H[i,0:] = np.array(data['H'])
				self.S[i,0:] = np.array(data['S'])
				self.state_parameter[i,0:] = np.array(data['state_parameter'])
				self.hydraulic_potential[i,0:] = np.array(data['hydraulic_potential']/1e6) #Given in MPa
				self.ExchangeTerm[i,0:] = np.array(data['ExchangeTerm']*secondyears) # given in m/yr
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
				self.U_SSA = np.append(self.U_SSA, data['U_SSA']*secondyears, axis=0)
				self.H = np.append(self.H, data['H'], axis=0)
				self.S = np.append(self.S, data['S'], axis=0)
				self.state_parameter = np.append(self.state_parameter, data['state_parameter'], axis=0)
				self.hydraulic_potential = np.append(self.hydraulic_potential, data['hydraulic_potential']/1e6, axis=0)
				self.ExchangeTerm = np.append(self.ExchangeTerm, data['ExchangeTerm']*secondyears, axis=0)
				self.t = np.append(self.t, data.get('t')/secondyears) #Given in years
			
			self.loadedIndices = newIndices + self.loadedIndices
			print('Loaded ' + str(np.size(newIndices)) + ' steps in time interval [' + str(t0) + ',' + str(self.t[-1]) + '] years')

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
		elif variable == 'effective_normal_stress':
			returnVal = self.rho*self.H*self.g/1e6 - (self.hydraulic_potential - self.water_density*self.g*self.b/1e6)
		elif variable == 'ExchangeTerm':
			returnVal = self.ExchangeTerm
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
		else:
			raise Exception('variable '+variable+' does not exist')
		return returnVal


	def plotPcolor(self,variable,printToFile = False, logscale = False, transpose = False, colorbar = False):
		# Plots spatiotemporal pcolor of given variable where the ice thickness is nonzero
		#plt.figure()
		var = self.getVariable(variable)
#		var[np.where(self.H<=0)]=float('NaN')
		var = np.ma.masked_array(var,self.H<=0)

		if transpose:
			xval = self.t
			yval = self.x
			var = var.transpose()

		else:
			xval = self.x
			yval = self.t
	
		if not logscale:
			pcol = plt.pcolormesh(xval,yval,var)
		else:
			pcol = plt.pcolormesh(xval,yval,var,norm=LogNorm())

		#pcol.set_rasterized(True)

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

