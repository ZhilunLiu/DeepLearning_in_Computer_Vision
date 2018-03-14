#!/usr/bin/env python3

import h5py
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm

if __name__ == "__main__":
    # Main code goes here


	with h5py.File('output.h5','r+') as fl:
   	## print("Keys: %s" % fl.keys())
		ls = list(fl.keys())

		# Get the data
		dataset1 = fl.get('dataset_1')
		data = np.array(dataset1)

		#taking mean along the last dimension
		data = data.mean(axis = -1)
		#print(data)

		#read arguement 
		for x in sys.argv:

			print ("The kernel sizes are: ", str(sys.argv[1:]))
		
		k1 = int(str(sys.argv[1]))
		k2 = int (str(sys.argv[2]))

		#filters
		from scipy import ndimage

		filtered_data1 = ndimage.filters.gaussian_filter(data, sigma=k1)
		filtered_data2 = ndimage.filters.gaussian_filter(data, sigma=k2)

		final_data = filtered_data2 - filtered_data1
		#normalization
		norm_data = np.abs(final_data-np.amin(final_data))/255
		from sklearn.preprocessing import normalize
		#norm_data = normalize(final_data, axis=0, norm='l1')
		##norm_data = np.linalg.norm(final_data, axis = -1)

		print(norm_data.shape)
		#plot
		plt.imshow(norm_data, cmap = cm.Greys_r)
		plt.show()



		data_file = h5py.File('filtered.h5', 'w')
		data_file.create_dataset('dataset_2', data=norm_data)
		data_file.close()





	  ##  exit(0)





