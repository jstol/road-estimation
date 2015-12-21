#!/usr/bin/env python2.7
# -*- coding: utf8 -*-
from __future__ import absolute_import, print_function, unicode_literals
"""Main script for training and evaluating models"""

import os
import numpy as np

def create_dir_if_not_exists(filename):
	if not os.path.exists(os.path.dirname(filename)):
		os.makedirs(os.path.dirname(filename))

# THINGS TO CONFIGURE
model = "neural_net"
num_superpixels = 5000
model_file = "best_model/{0}_{1}sp.npz.pkl".format(model, num_superpixels)
test_data_file = "extra_feature_matrices/test_examples_{0}sp.npz".format(num_superpixels)
test_prediction_dir = "final_predictions"
# -------------------

# Make any missing dirs
create_dir_if_not_exists(test_prediction_dir)

# Load in data
test_data = np.load(test_data_file)
test_X = test_data['inputs']

# Predict
test_predict = model.predict(test_X, model_file)

# Save
np.savez(test_prediction_file, predictions=test_pred)
