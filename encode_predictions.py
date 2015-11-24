#!/usr/bin/env python2.7
# -*- coding: utf8 -*-
from __future__ import absolute_import, print_function, unicode_literals
"""Script to encode road estimation predictions as a greyscale image"""

# Standard modules
import argparse
from os import path, makedirs
# Third party modules
from skimage.io import imread_collection, imread, imsave
import numpy as np

category = 'um'
file_type = '.png'
target_type = 'road'

# Set up command-line arguments
_argument_parser = argparse.ArgumentParser(
	prog='encode_predictions',
	description='Script to encode road estimation predictions as a greyscale image.'
)
# Define arguments
_argument_parser.add_argument('-m', '--map-input-file',
	dest='map_input_file',
	help='the location of the pixel map input file',
	required=False,
	default='pixelmaps.npz'
)
_argument_parser.add_argument('-i', '--predictions-input-path',
	dest='predictions_input_path',
	help='the location of the predictions .npz file',
	required=False,
	default='examples-predictions.npz'
)
_argument_parser.add_argument('-e', '--example-images-input-path',
	dest='example_images_input_path',
	help='the directory containing all corresponding examples images that predictions were made for',
	required=False,
	default='kit/data_road/training/image_2/'
)
_argument_parser.add_argument('-o', '--encoded-output-dir',
	dest='encoded_output_dir',
	help='directory where encoded predictions should be placed',
	required=False,
	default='predictions/encoded/'
)
_argument_parser.add_argument('-ov', '--overlay-output-dir',
	dest='encoded_overlay_output_dir',
	help='directory where encoded prediction overlays should be placed',
	required=False,
	default='predictions/encoded-overlay/'
)
args = _argument_parser.parse_args()

# Run script
if __name__ == '__main__':
	# Read in command line arguments
	map_input_file = args.map_input_file
	predictions_input_path = args.predictions_input_path
	example_images_input_path = args.example_images_input_path
	encoded_output_dir = args.encoded_output_dir
	encoded_overlay_output_dir = args.encoded_overlay_output_dir

	# Create any missing dirs
	if not path.exists(encoded_output_dir):
		makedirs(encoded_output_dir)
	if not path.exists(encoded_overlay_output_dir):
		makedirs(encoded_overlay_output_dir)

	# Read in pixel maps file
	pixel_maps = np.load(map_input_file)
	pixel_map_files = pixel_maps.files
	pixel_map_files.sort()

	all_predictions = np.load(predictions_input_path)['predictions']

	for file_id in pixel_map_files:
		image_id = file_id.split('{0}_'.format(category))[-1].split(file_type)[0] # ex. 000009

		print("Encoding image '{0}'".format(file_id))

		image = imread(path.join(example_images_input_path, "{0}{1}".format(file_id, file_type)))
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

		imsave(path.join(encoded_overlay_output_dir, "{0}_{1}_{2}{3}".format(category, target_type, image_id, file_type)), image)
		imsave(path.join(encoded_output_dir, "{0}_{1}_{2}{3}".format(category, target_type, image_id, file_type)), encoded_prediction_image) #Throws a warning?
