#!/usr/bin/env python2.7
# -*- coding: utf8 -*-
from __future__ import absolute_import, print_function, unicode_literals
"""Script to compute superpixels (using SLICO) for all provided images"""

# Standard modules
import argparse
import os.path as path
# Third party modules
from skimage.segmentation import slic
from skimage.io import imread_collection, imread
import numpy as np

import matplotlib.pyplot as plot

_NUM_FEATURES = 7 # Avg R,G,B, delta_x, delta_y, average x, average y

category = 'um'
file_type = '.png'
target_type = 'road'

# Read in command line arguments
_argument_parser = argparse.ArgumentParser(
	prog='precompute',
	description='Script to compute superpixel maps (using SLICO) for the provided directory of images. Outputs an .npz file containing a superpixel mapping matrix for each image.'
)
# Define arguments
_argument_parser.add_argument('-i', '--examples-input-path',
	dest='examples_input_path',
	help='the path pattern to match when importing images, or a list of file paths',
	nargs='+',
	required=True
)
_argument_parser.add_argument('-t', '--targets-input-directory',
	dest='targets_input_dir',
	help='the directory containing all target images (OPTIONAL - only used when training)',
	required=False,
	default=None
)
_argument_parser.add_argument('-m', '--map-output-file',
	dest='map_output_file',
	help='the location of the pixel map output file',
	required=False,
	default='pixelmaps.npz'
)
_argument_parser.add_argument('-e', '--examples-output-file',
	dest='examples_output_file',
	help='the location of the examples/targets output file',
	required=False,
	default='examples.npz'
)
_argument_parser.add_argument('-n', '--num-superpixels',
	type=int,
	dest='num_superpixels',
	help='the (approximate) number of super pixels to segment each image into',
	required=False,
	default=100
)
args = _argument_parser.parse_args()

# Run app
if __name__ == '__main__':
	example_input_path = args.examples_input_path
	targets_input_dir = args.targets_input_dir
	generate_targets = targets_input_dir is not None
	num_superpixels = args.num_superpixels
	map_output_file = args.map_output_file
	examples_output_file = args.examples_output_file

	print("Outputting pixel maps file to: '{0}'".format(map_output_file))

	# Read in the training images
	training_images = imread_collection(example_input_path)
	# Initialize required variables: superpixel mapping, input matrix, and targets (which only gets used if needed)
	superpixel_maps = {}
	inputs = np.empty((0,_NUM_FEATURES), float)
	targets = np.empty((0,1), int)

	for i, image in enumerate(training_images):
		example_name = path.splitext(path.basename(training_images.files[i]))[0] # ex. um_000009
		
		print("Processing {0}".format(example_name))

		# Record this superpixels mapping		
		superpixels = slic(image, n_segments=num_superpixels, slic_zero=True)
		superpixel_maps[example_name] = superpixels

		# Read in the ground truth image if we're also computing target labels
		if generate_targets:
			# Generate a pattern that matches the ground truth
			image_id = example_name.split('{0}_'.format(category))[-1].split(file_type)[0] # ex. 000009
			target_filename = "{0}_{1}_{2}{3}".format(category, target_type, image_id, file_type) # ex. um_road_000009.png
			# Load the image
			target_image = imread(path.join(targets_input_dir, target_filename))
			# The blue channel holds the truth
			target_image = target_image[:,:,2] > 0

		# Compute the examples matrix
		superpixels_count = len(np.unique(superpixels))
		for sp in xrange(superpixels_count):
			# Get a mask for this specific superpixel
			superpixel = (superpixels == sp)
			# Compute position-related features
			x, y = superpixel.nonzero()
			x_avg = np.average(x)
			y_avg = np.average(y)
			x_diff = np.max(x)-np.min(x)
			y_diff = np.max(y)-np.min(y)
			# Compute RGB-related features
			r_avg = np.average(image[superpixel, 0]) # R
			g_avg = np.average(image[superpixel, 1]) # G
			b_avg = np.average(image[superpixel, 2]) # B

			inputs = np.vstack((inputs, np.array([x_avg, y_avg, x_diff, y_diff, r_avg, g_avg, b_avg])))

			# Figure out targets at the super-pixel level
			if generate_targets:
				# Compute the average classification for the superpixel
				superpixel_target = np.count_nonzero(target_image[superpixel])/float(np.count_nonzero(superpixel)) >= 0.5
				targets = np.vstack((targets, np.array([superpixel_target])))

	# Save the superpixel mappings
	np.savez(map_output_file, **superpixel_maps)
	# Save the feature matrix (and target, if necessary)
	examples = {'inputs': inputs}
	if generate_targets:
		examples['targets'] = targets
	np.savez(examples_output_file, **examples)

	# a = np.array([[10,20,30],[40,60,80],[100,120,140]])
	# superpixels_map = np.array([[1,1,2],[1,2,2],[3,3,4]])
	# a_superpixels_map = ma.array(a, mask=(superpixels_map != 2))

	# i = imread('kit/data_road/training/image_2/um_000000.png')
	# supers = slic(i, slic_zero=True, n_segments=2)
	# super_pixel_0 = (supers == 0)
	# i[super_pixel_0, 0].shape

	# x,y = super_pixel_0.nonzero()
	# np.average(x)
	# np.max(x)-np.min(x)

	#(map >= 78) & (map <= 81)
