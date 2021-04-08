import numpy as np
import scipy.io as sio
import json
import sys
import os
import tables as tb


class Output():
	
	def __copy__(self):
		pass

	def __deepcopy__(self, memo): # don't copy output object (deepcopy is needed for adaptive error estimates)
		pass 

	def __init__(self, output_interval = 1, foldername = 'results', file_format = 'mat', flush_interval = 1000, model = None, reduced = False, reduced_output_interval = 1):

		self.ind = 0
		self.output_interval = output_interval
		self.file_format = file_format
		self.foldername = foldername
		self.filename = sys.argv[0]
		self.model = model
		self.reduced = reduced
		self.reduced_output_interval = reduced_output_interval
		self.flush_interval = flush_interval

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

			if self.reduced and self.file_format is not 'hdf5':
				os.mkdir(self.foldername+'/data_reduced')

			# Create dictionary with initial conditions as defined in the different classes
			varDictionary_INIT = self.getDictionary(init=True)

			if self.file_format == 'mat':
				sio.savemat(self.foldername+'/src/INIT.mat', varDictionary_INIT)
			elif self.file_format == 'json':
				with open (self.foldername+'/src/INIT.json', 'w') as file:
					json.dump(varDictionary_INIT,file)
			elif self.file_format=='hdf5':
				self.output_file = tb.open_file(self.foldername+'/data/out.h5','w')
				output_init_group = self.output_file.create_group(self.output_file.root,'INIT','fixed quantities') # Create average variables group
				for key in varDictionary_INIT:
					self.output_file.create_array(output_init_group, str(key), np.array(varDictionary_INIT[key]))

				output_group = self.output_file.create_group(self.output_file.root,'full','spatial variables across the full system') # Create average variables group
				output_average_group = self.output_file.create_group(self.output_file.root,'average','variables averaged over glacier length') # Create average variables group

				varDictionary = self.getDictionary(init=False)
				for key in varDictionary:
					if np.size(varDictionary[key])>1:
						self.output_file.create_earray(output_group, str(key), tb.FloatAtom(), (0,np.size(varDictionary[key]),) )
					else:
						self.output_file.create_earray(output_group, str(key), tb.FloatAtom(), (0,) )
					self.output_file.create_earray(output_average_group, str(key), tb.FloatAtom(), (0,) )
				self.output_file.flush()

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
			elif self.file_format == 'hdf5':
				for key in varDictionary:
					self.output_file.root.full[key].append(np.array([varDictionary[key],]))

				#self.output_file.flush()
		
		if(self.reduced and self.ind%self.reduced_output_interval==0): #Output average quantities
			varDictionary = self.getDictionary(init = False)
			averageVarDictionary = self.getAverageDictionary(varDictionary)
			if self.file_format == 'mat':
				sio.savemat(self.foldername+'/data_reduced/'+str(self.ind)+'.mat', averageVarDictionary)
			elif self.file_format == 'json':
				with open (self.foldername+'/data_reduced/'+str(self.ind)+'.json', 'w') as file:
					json.dump(averageVarDictionary,file)
			
			elif self.file_format == 'hdf5':
				for key in varDictionary:
					self.output_file.root.average[key].append(np.array([averageVarDictionary[key],]))
				

		if(self.file_format == 'hdf5' and self.ind%self.flush_interval==0):
			self.output_file.flush()

		self.ind = self.ind+1

	def closeFile(self):
		if self.file_format=='hdf5':
			self.output_file.close()
		else:
			pass


	def getAverageDictionary(self,dictionary): # return dictionary with all values in dictionary averaged over the glacier length
		zeroThickness = self.model.H<=0
		averageVarDictionary = dictionary
		for key in averageVarDictionary:
			if isinstance(averageVarDictionary[key],list) and len(averageVarDictionary[key]) == len(zeroThickness): # take average of quantities defined across the entire domain
				var = np.asarray(averageVarDictionary[key])
				var[zeroThickness]=float('NaN')
				var = np.nanmean(var)
				averageVarDictionary[key] = var

		return averageVarDictionary


	def getMaxDictionary(self,dictionary): # return dictionary with maximum of all values in dictionary over the glacier length
		zeroThickness = self.model.H<=0
		maxVarDictionary = dictionary
		for key in maxVarDictionary:
			if isinstance(maxVarDictionary[key],list) and len(maxVarDictionary[key]) == len(zeroThickness): # take average of quantities defined across the entire domain
				var = np.asarray(maxVarDictionary[key])
				var[zeroThickness]=float('NaN')
				var = np.nanmax(var)
				maxVarDictionary[key] = var

		return maxVarDictionary


	def getMinDictionary(self,dictionary):# return dictionary with minimum of all values in dictionary over the glacier length
		zeroThickness = self.model.H<=0
		minVarDictionary = dictionary
		for key in minVarDictionary:
			if isinstance(minVarDictionary[key],list) and len(minVarDictionary[key]) == len(zeroThickness): # take average of quantities defined across the entire domain
				var = np.asarray(maxVarDictionary[key])
				var[zeroThickness]=float('NaN')
				var = np.nanmin(var)
				minVarDictionary[key] = var

		return minVarDictionary


	def getDictionary(self,init=False):
		varDictionary = self.model.getDictionary(init = init)
		if(self.model.FrictionLaw is not None):
			varDictionary.update(self.model.FrictionLaw.getDictionary(init = init))
		if(self.model.DrainageSystem is not None):
			varDictionary.update(self.model.DrainageSystem.getDictionary(init = init))
		return varDictionary

		
