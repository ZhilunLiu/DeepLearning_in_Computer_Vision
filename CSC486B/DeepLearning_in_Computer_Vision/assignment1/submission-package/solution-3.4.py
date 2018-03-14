#!/usr/bin/env python3

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

if __name__ == "__main__":
	# Main code goes here
	with h5py.File('filtered.h5', 'r+') as fl:
		ls = list(fl.keys())
		#Get the data
		dataset = fl.get('dataset_2')
		data = np.array(dataset)

		sorted_data = np.argsort(data)
		threshholdIndex = int(len(sorted_data)*0.95)
		#print(threshholdIndex)
		threshhold = sorted_data[threshholdIndex]



		#thdata = np.copy(data)
		#thdata = np.full_like(data,0)

		for i in range(len(data)):
			for j in range(len(data[i])):
				print(threshhold)
				if data[i][j] < threshhold:
					data[i][j] = 0


		#get the index of the top 5% item
		#ind = np.argsort(data)
		#ind2 = ind[0:32]
		#ind = np.argpartition(data,-5)[-5:]
		#create a new array with same size 
		#thdata = np.copy(data)
		#thdata = np.full_like(data,0)
		#copy the top 5% item into the new array
		#thdata[ind2] = data[ind2]
		print(thdata)
		print(ind.shape)
		print(np.count_nonzero(thdata))

		
		plt.imshow(thdata)
		plt.show


		data_file = h5py.File('threshhold.h5', 'w')
		data_file.create_dataset('dataset_3', data=thdata)
		data_file.close()

  ##  exit(0)
