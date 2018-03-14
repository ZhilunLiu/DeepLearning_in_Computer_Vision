#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import h5py


if __name__ == "__main__":
    # Main code goes here
	from PIL import Image
	img = Image.open("input.jpg")
	pix  = np.asarray(img)
	print(pix)
	
	img=mpimg.imread('input.jpg')
	imgplot = plt.imshow(img)
	plt.show()

	h5f = h5py.File('output.h5', 'w')
	h5f.create_dataset('dataset_1', data=pix)
	h5f.close()


##To Invert the Image, Plz uncomment the code below 


##	pix = 255 - pix
##	print(pix)

##	new_img = Image.fromarray(np.uint8(pix))
##	new_img.save("F2.jpg")

##	img=mpimg.imread('F2.jpg')
##	imgplot = plt.imshow(img)
##	plt.show()


	

    #exit(0)
