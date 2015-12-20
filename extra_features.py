#!/usr/bin/env python2.7
# -*- coding: utf8 -*-
from __future__ import absolute_import, print_function, unicode_literals
"""Script to encode road estimation predictions as a greyscale image"""

# Standard modules
import argparse
from os import path, makedirs
# Third party modules
from skimage.io import imread, imsave
from skimage.color import rgb2hsv, rgb2grey
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import exposure, feature
import numpy as np

file_type = '.png'

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
_argument_parser.add_argument('-i', '--example-matrices-input-path',
	dest='example_matrices_input_path',
	help='the location of the example matrices .npz file',
	required=False,
	default='examples-predictions.npz'
)
_argument_parser.add_argument('-e', '--example-images-input-path',
	dest='example_images_input_path',
	help='the directory containing all corresponding examples images that predictions were made for',
	required=False,
	default='kit/data_road/training/image_2/'
)
_argument_parser.add_argument('-o', '--augmented-example-matrices-output-dir',
	dest='augmented_example_matrices_output_dir',
	help='directory where augmented example matrices should be placed',
	required=False,
	default='predictions/encoded/'
)
args = _argument_parser.parse_args()

# Run script
if __name__ == '__main__':
	# Read in command line arguments
	map_input_file = args.map_input_file
	example_matrices_input_path = args.example_matrices_input_path
	example_images_input_path = args.example_images_input_path
	augmented_example_matrices_output_dir = args.augmented_example_matrices_output_dir

	# Create any missing dirs
	if not path.exists(augmented_example_matrices_output_dir):
		makedirs(augmented_example_matrices_output_dir)

	# Read in pixel maps file
	pixel_maps = np.load(map_input_file)
	pixel_map_files = pixel_maps.files
	pixel_map_files.sort()

	# Read in original example matrices file
	examples = np.load(example_matrices_input_path)
	example_inputs = examples['inputs']

	# Make an empty matrix for the new features
	new_example_inputs = np.empty((0, 8))

	# Calculate new features
	for file_id in pixel_map_files:
		category = file_id.split('_')[0]
		image_id = file_id.split('{0}_'.format(category))[-1].split(file_type)[0] # ex. 000009

		print("Working on image '{0}'".format(file_id))

		image = imread(path.join(example_images_input_path, "{0}{1}".format(file_id, file_type)))

		superpixels_mask = pixel_maps[file_id]
		num_superpixels = len(np.unique(superpixels_mask))

		r = image[:,:,0]
		g = image[:,:,1]
		b = image[:,:,2]

		# Calculate new features
		# HSV
		hsv_image = rgb2hsv(image)
		h = hsv_image[:,:,0]
		s = hsv_image[:,:,1]
		v = hsv_image[:,:,2]
		# Greyscale - used for some calculations
		grey_image = rgb2grey(image)
		# Equalize
		grey_image_global_equalize = exposure.equalize_hist(grey_image)

		# Entropy
		image_entropy = entropy(grey_image_global_equalize, disk(5))
		# Edges
		image_edges = feature.canny(grey_image_global_equalize)

		# Aggregate the new features, per superpixel
		for superpixel_i in xrange(num_superpixels):
			# Get a mask for this specific superpixel
			superpixel = (superpixels_mask == superpixel_i)
			pixel_count = np.sum(superpixel)
			# Compute RGB-related features
			r_sp = image[superpixel, 0]
			g_sp = image[superpixel, 1]
			b_sp = image[superpixel, 2]
			r_avg = np.average(r) # R
			g_avg = np.average(g) # G
			b_avg = np.average(b) # B
			# Variance
			r_var = ( np.sum(np.square(r))/pixel_count ) - (r_avg*r_avg)
			g_var = ( np.sum(np.square(g))/pixel_count ) - (g_avg*g_avg)
			b_var = ( np.sum(np.square(b))/pixel_count ) - (b_avg*b_avg)
			# HSV
			h_avg = np.average(hsv_image[superpixel, 0])
			s_avg = np.average(hsv_image[superpixel, 1])
			v_avg = np.average(hsv_image[superpixel, 2])
			# Entropy / Edge
			ent = np.average(image_entropy[superpixel])
			edg = np.average(image_edges[superpixel])

			new_example_inputs = np.vstack((new_example_inputs, np.array([r_var, g_var, b_var, h_avg, s_avg, v_avg, ent, edg])))

	# Save the feature matrix (and target, if necessary)
	new_output = {'inputs': np.hstack((example_inputs, new_example_inputs))}
	if 'targets' in examples:
		new_output['targets'] = examples['targets']
	
	np.savez(path.join( augmented_example_matrices_output_dir, path.basename(example_matrices_input_path) ), **new_output)	
