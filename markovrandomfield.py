#!/usr/bin/env python2.7
# -*- coding: utf8 -*-
from __future__ import absolute_import, print_function, unicode_literals
"""defines classes that encapsulates the sci-kit learn ML algorithms"""

#general
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.externals import joblib
import math

class pixelmap:
	"""
	Interface to pixelmap:

	pixelmap_a = pixelmap()	:							create a new instance of pixelmap
	pixelmap.set_conn_energy :							set the strength of the connections between pixels - 
														this impact the relative importance of keeping the labels smooth vs. 
														being consistent with the superpixel prediction
	pixelmap.load_superpixel_classifier_predictions :	load pixel level predictions from superpixel classifier, 
														need to specify height and width of image
	pixelmap.mcmc_update:								perform 1 iteration of mcmc (iterate through every pixel), need to specify temperature

	pixelmap.eval_energy:								compute the energy of the current state of pixel labels

	"""
	def __init__(self):
		self.height = None
		self.width = None
		self.pixel_labels = None #this is going to be a D dimensional array (each element is the label for 1 pixel) storing the predicted label
		self.pixel_conns = None #this is going to be a sparse DxD dimensional matrix (i,j)th element is the connection between pixel i and pixel j
		self.pixel_priors = None #this is going to be a D dimensional array (each element is the label for 1 pixel) storing the predicted probability of the pixel being road based on superpixel classifier
		self.conn_energy = None #we need to define this somewhere

	def set_conn_energy(self, conn_energy):
		self.conn_energy = conn_energy

	def load_superpixel_classifier_predictions(self, pixel_lvl_predictions, height, width):

		pixel_lvl_predictions = pixel_lvl_predictions.ravel()
		
		self.height = height
		self.width = width
		#assign labels
		self.pixel_labels = np.zeros(height*width)
		self.pixel_labels[pixel_lvl_predictions>=0.5] = 1
		
		#load the pixel priors
		self.pixel_priors = np.zeros(height*width)
		self.pixel_priors = pixel_lvl_predictions

	def pixel_index_to_posn(self, pixel_idx):
		y_pos = math.floor(pixel_idx/float(self.width))
		x_pos = pixel_idx - y_pos*self.width

		return (x_pos, y_pos)

	def pixel_posn_to_idx(self, x_pos, y_pos):
		pixel_index = y_pos*self.width+x_pos

		return pixel_index

	def neighbours_list(self, pixel_idx):
		(x_pos, y_pos) = self.pixel_index_to_posn(pixel_idx)


		neighbours_index_list = [];
		neighbours_label_list = [];

		#if the pixel is in the top row, there is no top neighbour; otherwise
		if (y_pos > 0):
			top_neighbour_idx = self.pixel_posn_to_idx(x_pos, y_pos-1)
			neighbours_index_list.append(top_neighbour_idx)
			neighbours_label_list.append(self.pixel_labels[top_neighbour_idx])

		#if the pixel is in the bottom row, there is no bottom neighbour; otherwise
		if (y_pos < self.height-1):
			bottom_neighbour_idx = self.pixel_posn_to_idx(x_pos, y_pos+1)
			neighbours_index_list.append(bottom_neighbour_idx)
			neighbours_label_list.append(self.pixel_labels[bottom_neighbour_idx])

		#if the pixel is in the left column, there is no left neighbour; otherwise
		if (x_pos > 0):
			left_neighbour_idx = self.pixel_posn_to_idx(x_pos-1, y_pos)
			neighbours_index_list.append(left_neighbour_idx)
			neighbours_label_list.append(self.pixel_labels[left_neighbour_idx])

		#if the pixel is in the right column, there is no right neighbour; otherwise
		if (x_pos < self.width-1):
			right_neighbour_idx = self.pixel_posn_to_idx(x_pos+1, y_pos)
			neighbours_index_list.append(right_neighbour_idx)
			neighbours_label_list.append(self.pixel_labels[right_neighbour_idx])

		return (neighbours_index_list, neighbours_label_list)

	def eval_energy(self):
		#compute the global component of energy
		global_energy = 0

		for pixel_idx in xrange(self.height*self.width):
			pixel_label = self.pixel_labels[pixel_idx]
			(neighbours_index_list, neighbours_label_list) = self.neighbours_list(pixel_idx)
			for neighbour_label in neighbours_label_list:
				global_energy = global_energy + (-1)*(2*pixel_label-1)*(2*neighbour_label-1)*self.conn_energy

		#compute individual pixel component of energy
		pixel_labels = self.pixel_labels
		pixel_priors = self.pixel_priors

		pixel_priors_t1 = 0.5*np.ones(pixel_priors.shape)
		pixel_priors_t0 = 0.5*np.ones(pixel_priors.shape)
		pixel_priors_t1[pixel_labels == 1] = pixel_priors[pixel_labels == 1]
		pixel_priors_t0[pixel_labels == 0] = pixel_priors[pixel_labels == 0]
		sum_pixel_lvl_energy = -(np.dot(pixel_labels, np.log(pixel_priors_t1)) + np.dot((1-pixel_labels), np.log(1- pixel_priors_t0)))

		total_energy = global_energy+sum_pixel_lvl_energy

		return total_energy

	def one_flip_energy_change(self, pixel_idx):
		pixel_label = self.pixel_labels[pixel_idx]
		proposed_pixel_label = 1- pixel_label
		pixel_lvl_prior = self.pixel_priors[pixel_idx]

		#compute the change in global component of energy
		old_global_energy = 0
		new_global_energy = 0
		(neighbours_index_list, neighbours_label_list) = self.neighbours_list(pixel_idx)

		#compute current energy
		for neighbour_label in neighbours_label_list:
			old_global_energy = old_global_energy + (-1)*(2*pixel_label-1)*(2*neighbour_label-1)*self.conn_energy

		#compute proposed energy
		for neighbour_label in neighbours_label_list:
			new_global_energy = new_global_energy + (-1)*(2*proposed_pixel_label-1)*(2*neighbour_label-1)*self.conn_energy

		#compute change
		global_energy_change = new_global_energy - old_global_energy

		#compute the change in local component of energy
		if pixel_label == 1:
			old_local_energy = (-1)*np.log(pixel_lvl_prior)
			new_local_energy = (-1)*np.log(1-pixel_lvl_prior)
		elif pixel_label == 0:
			old_local_energy = (-1)*np.log(1-pixel_lvl_prior)
			new_local_energy = (-1)*np.log(pixel_lvl_prior)


		#compute change
		local_energy_change = new_local_energy - old_global_energy

		#compute total change in energy
		total_energy_change = global_energy_change+local_energy_change

		return total_energy_change

	def mcmc_one_flip(self, pixel_idx, temperature):
		energy_change = self.one_flip_energy_change(pixel_idx)
		prob_flip = min(1, np.exp(-temperature*energy_change))

		#pick a random number uniformly from 0 to 1, if the random number < prob_flip, then flip;
		#otherwise don't flip
		rand_num = np.random.rand()

		if rand_num < prob_flip:
			self.pixel_labels[pixel_idx] = 1-self.pixel_labels[pixel_idx]

	def mcmc_update(self, temperature):
		for pixel_idx in xrange(self.height*self.width):
			self.mcmc_one_flip(pixel_idx, temperature)





print('==============================')
print('Testing')
print('==============================')

#build test priors
image_pixel_priors = np.ones([200, 400])*0.1
image_pixel_priors[:, 120:180] = 0.8
image_pixel_priors_flat = image_pixel_priors.ravel()

predicted_labels = pixelmap()

predicted_labels.load_superpixel_classifier_predictions(image_pixel_priors_flat, 200, 400)

predicted_labels.set_conn_energy(0.1) #this is required to set the strength of connections ()

print('Initial energy:')
print(predicted_labels.eval_energy())

print('MCMC update 1')
predicted_labels.mcmc_update(0.1)
print(predicted_labels.eval_energy())

print('MCMC update 2')
predicted_labels.mcmc_update(1)
predicted_labels.mcmc_update(1)
predicted_labels.mcmc_update(1)
predicted_labels.mcmc_update(1)
predicted_labels.mcmc_update(1)
predicted_labels.mcmc_update(1)
print(predicted_labels.eval_energy())

print('MCMC update 3')
predicted_labels.mcmc_update(10)
predicted_labels.mcmc_update(10)
predicted_labels.mcmc_update(10)
predicted_labels.mcmc_update(10)
predicted_labels.mcmc_update(10)
predicted_labels.mcmc_update(10)
print(predicted_labels.eval_energy())

print('MCMC update 4')
predicted_labels.mcmc_update(13)
predicted_labels.mcmc_update(13)
predicted_labels.mcmc_update(13)
predicted_labels.mcmc_update(13)
predicted_labels.mcmc_update(13)
predicted_labels.mcmc_update(13)
print(predicted_labels.eval_energy())

print('MCMC update 5')
predicted_labels.mcmc_update(15)
predicted_labels.mcmc_update(15)
predicted_labels.mcmc_update(15)
predicted_labels.mcmc_update(15)
predicted_labels.mcmc_update(15)
predicted_labels.mcmc_update(15)
print(predicted_labels.eval_energy())

print('MCMC update 6')
predicted_labels.mcmc_update(17)
predicted_labels.mcmc_update(17)
predicted_labels.mcmc_update(17)
predicted_labels.mcmc_update(17)
predicted_labels.mcmc_update(17)
predicted_labels.mcmc_update(17)
print(predicted_labels.eval_energy())


print('MCMC update 7')
predicted_labels.mcmc_update(20)
predicted_labels.mcmc_update(20)
predicted_labels.mcmc_update(20)
predicted_labels.mcmc_update(20)
predicted_labels.mcmc_update(20)
predicted_labels.mcmc_update(20)
print(predicted_labels.eval_energy())














