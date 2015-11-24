# import matplotlib.pyplot as plt
from skimage.io import imread_collection, imread, imsave
import numpy as np

pixel_maps = np.load('pixelmaps.npz')
pixel_map_files = pixel_maps.files
pixel_map_files.sort()

all_predictions = np.load('examples-predictions.npz')['predictions']

for file_id in pixel_map_files:
	print("Encoding image '{0}'".format(file_id))

	image = imread('kit/data_road/training/image_2/{0}.png'.format(file_id))
	encoded_prediction_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.float64)

	superpixels_mask = pixel_maps[file_id]
	num_superpixels = len(np.unique(superpixels_mask))

	image_predictions = all_predictions[ : num_superpixels]
	all_predictions = all_predictions[num_superpixels : ]

	for superpixel_i in xrange(num_superpixels):
		# Color the road red for visualization of the prediction
		if image_predictions[superpixel_i] >= 0.5:
			image[superpixels_mask == superpixel_i, 1] = image_predictions[superpixel_i]
			image[superpixels_mask == superpixel_i, 2] = image_predictions[superpixel_i]
		# Encode the prediction in another image
		encoded_prediction_image[superpixels_mask == superpixel_i] = image_predictions[superpixel_i]

	imsave("overlayed_predictions/{0}.png".format(file_id), image)
	imsave("predictions/{0}.png".format(file_id), encoded_prediction_image) #Throws a warning?
