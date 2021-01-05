import numpy as np
import scipy.io as sio
import json
import sys
import os

class Output():
	def __init__(self, output_interval = 1, foldername = 'results', file_format = 'mat', model = None, reduced = False, reduced_output_interval = 1):

		self.ind = 0
		self.output_interval = output_interval
		self.file_format = file_format
		self.foldername = foldername
		self.filename = sys.argv[0]
		self.model = model
		self.reduced = reduced
		self.reduced_output_interval = reduced_output_interval

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

			if self.reduced:
				os.mkdir(self.foldername+'/data_reduced')

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

		if(self.ind%self.output_interval==0): #Output full data
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

		
		if(self.reduced and self.ind%self.reduced_output_interval==0): #Output average quantities
			varDictionary = model.getDictionary(init = False)
			if(model.FrictionLaw is not None):
				varDictionary.update(model.FrictionLaw.getDictionary(init = False))
			if(model.DrainageSystem is not None):
				varDictionary.update(model.DrainageSystem.getDictionary(init = False))
			
			averageVarDictionary = self.getAverageDictionary(varDictionary)
			if self.file_format == 'mat':
				sio.savemat(self.foldername+'/data_reduced/'+str(self.ind)+'.mat', averageVarDictionary)
			elif self.file_format == 'json':
				with open (self.foldername+'/data_reduced/'+str(self.ind)+'.json', 'w') as file:
					json.dump(averageVarDictionary,file)

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

	def getAverageDictionary(self,dictionary):
		zeroThickness = self.model.H<=0
		averageVarDictionary = dictionary
		for key in averageVarDictionary:
			if isinstance(averageVarDictionary[key],list) and len(averageVarDictionary[key]) == len(zeroThickness): # take average of quantities defined across the entire domain
				var = np.asarray(averageVarDictionary[key])
				var[zeroThickness]=float('NaN')
				var = np.nanmean(var)
				averageVarDictionary[key] = var

		return averageVarDictionary
		
