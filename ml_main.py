#!/usr/bin/env python2.7
# -*- coding: utf8 -*-
from __future__ import absolute_import, print_function, unicode_literals
"""Main script for training and evaluating models"""

# Standard modules
import argparse
# Third party modules
import numpy as np
import matplotlib.pyplot as plt
# Model implementations
from models import KNNModel, SVMModel, LogisticRegressionModel, NeuralNetworkModel, MOGModel, DecisionTreeModel, AdaboostModel, BaggingModel
from sklearn.ensemble import ExtraTreesClassifier

def run_feature_importance():
# Feature importance forest
# Build a forest and compute the feature importances
	print('Feature importance using random forest')

	# TODO also look at RandomForestClassifier - what's the difference?
	forest = ExtraTreesClassifier(n_estimators=250, random_state=0)

	forest.fit(train_X, train_y)
	importances = forest.feature_importances_
	std = np.std([rand_tree.feature_importances_ for rand_tree in forest.estimators_], axis=0)
	indices = np.argsort(importances)[::-1]

	# Print the feature ranking
	print("Feature ranking:")

	for f in range(train_X.shape[1]):
	    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

	# Plot the feature importances of the forest
	plt.figure()
	plt.title("Feature importances")
	plt.bar(range(train_X.shape[1]), importances[indices],
	       color="r", yerr=std[indices], align="center")
	plt.xticks(range(train_X.shape[1]), indices)
	plt.xlim([-1, train_X.shape[1]])
	plt.show()

# Global vars
available_models = ['knn', 'logistic', 'svm', 'mog', 'neural_net', 'decision_tree']
available_ensemble_methods = ['adaboost', 'bagging']

# Set up command-line arguments
_argument_parser = argparse.ArgumentParser(
	prog='ml_main.py',
	description='Main script for training and evaluating models.'
)
# Define arguments
_argument_parser.add_argument('-m', '--model',
	choices=available_models,
	dest='model_name',
	help='the model to run',
	required=True
)
_argument_parser.add_argument('-e', '--ensemble-type',
	choices=available_ensemble_methods,
	dest='ensemble_method',
	help='the ensemble method to use (if any)',
	required=False,
	default=None
)
# TODO add arguments for file locations, etc.
args = _argument_parser.parse_args()

# Run script
if __name__ == '__main__':
	model_name = args.model_name
	ensemble_method = args.ensemble_method

	training_data_file = 'train_examples_5000sp.npz'
	valid_data_file = 'valid_examples_5000sp.npz'
	test_data_file = 'test_examples_5000sp.npz'

	if not ensemble_method:
		print("Running {0}".format(model_name))
	else:
		print("Running {0} ({1})".format(ensemble_method, model_name))

	#training set
	train_data = np.load(training_data_file)
	train_X = train_data['inputs']
	train_y = train_data['targets']

	#Compute mean and range of X values in order to normalize training, validation and test
	train_X_features_mean = np.zeros(train_X.shape)
	train_X_features_range = np.zeros(train_X.shape)

	for j in xrange(train_X.shape[1]):
		train_X_features_mean[:, j] = float(np.mean(train_X[:, j]))
		train_X_features_range[:, j]  = float((np.amax(train_X[:, j])-np.amin(train_X[:, j])))

	#normalize the value of features to be between -1 to 1
	for j in xrange(train_X.shape[1]):
		train_X[:,j] = (train_X[:,j] - train_X_features_mean[0,j])/train_X_features_range[0,j]

	#validation data
	valid_data = np.load(valid_data_file)
	valid_X = valid_data['inputs']
	valid_y = valid_data['targets']

	#normalize the value of features to be between -1 to 1
	for j in xrange(valid_X.shape[1]):
		valid_X[:,j] = (valid_X[:,j] - train_X_features_mean[0,j])/train_X_features_range[0,j]

	# TODO write code to evaluate predictions on the "test" data

	# #test data
	# test_data = np.load(test_data_file)
	# test_X = valid_data['inputs']
	# test_y = valid_data['targets']

	# #normalize the value of features to be between -1 to 1
	# for j in xrange(test_X.shape[1]):
	# 	test_X[:,j] = (test_X[:,j] - train_X_features_mean[0,j])/train_X_features_range[0,j]

	# Run classifier
	print('''	
		==================================
		Running Classifier...
		==================================
	''')

	model = None
	model_file = None
	train_output_file = None
	valid_output_file = None

	# Standard algorithms
	if not ensemble_method:
		# KNN
		if model_name == 'knn':
			model = KNNModel({'k': 10})
			model_file = 'knn_test_model.npz'
			train_output_file = 'examples-train-predictions-knn-5000.npz'
			valid_output_file = 'examples-valid-predictions-knn-5000.npz'

		# Logistic
		elif model_name == 'logistic':
			model = LogisticRegressionModel({'penalty': 'l2', 'regularization_term': 0.1})
			model_file = 'logistic_test_model.npz'
			train_output_file = 'examples-train-predictions-logistic-5000.npz'
			valid_output_file = 'examples-valid-predictions-logistic-5000.npz'

		# SVM
		elif model_name == 'svm':
			model = SVMModel({'kernel': 'rbf', 'probability_flag': False})
			model_file = 'svm_test_model.npz'
			train_output_file = 'examples-train-predictions-svm.npz'
			valid_output_file = 'examples-valid-predictions-svm.npz'

		# MoG
		elif model_name == 'mog':
			model = MOGModel({'n_components': 20})
			model_file = 'mog_test_model.npz'
			train_output_file = 'examples-train-predictions-mog-5000.npz'
			valid_output_file = 'examples-valid-predictions-mog-5000.npz'

		# Multi-layer perceptron (NNets)
		elif model_name == 'neural_net':
			model = NeuralNetworkModel({'xxx': 0})
			model_file = 'mlp_test_model.npz'
			train_output_file = 'examples-train-predictions-mlp.npz'
			valid_output_file = 'examples-valid-predictions-mlp.npz'

		# Decision tree
		elif model_name == 'decision_tree':
			model = DecisionTreeModel({'criterion': 'gini', 'max_depth': 10, 'min_samples_split': 5})
			model_file = 'decision_tree_test_model.npz'
			train_output_file = 'examples-train-predictions-decisiontree-5000.npz'
			valid_output_file = 'examples-valid-predictions-decisiontree-5000.npz'

		else:
			raise NotImplementedError, "Invalid model name"

	# Adaboost
	elif (ensemble_method == 'adaboost'):
		# Adaboost decision tree
		if model_name == 'decision_tree':
			model = AdaboostModel({'algorithm_name': 'decision_tree', 'n_estimators': 25, 'criterion': 'gini', 'max_depth': 1, 'min_samples_split': 5})
			model_file = 'ada_dt_test_model.npz'
			train_output_file = 'examples-train-predictions-ada_dt-5000.npz'
			valid_output_file = 'examples-valid-predictions-ada_dt-5000.npz'

		else:
			raise NotImplementedError, "Invalid adaboost model"

	# Bagging
	elif (ensemble_method == "bagging"):
		# Bagging decision tree
		if model_name == 'decision_tree':
			model = BaggingModel({'algorithm_name': 'decision_tree', 'n_estimators': 25, 'criterion': 'gini', 'max_depth': 3, 'min_samples_split': 5})
			model_file = 'bag_dt_test_model.npz'
			train_output_file = 'examples-train-predictions-bag_dt-5000.npz'
			valid_output_file = 'examples-valid-predictions-bag_dt-5000.npz'

		elif model_name == 'logistic':
			model = BaggingModel({'algorithm_name': 'logistic', 'n_estimators': 25, 'penalty': 'l2', 'regularization_term': 0.01})
			model_file = 'bag_log_test_model.npz'
			train_output_file = 'examples-train-predictions-bag_log-5000.npz'
			valid_output_file = 'examples-valid-predictions-bag_log-5000.npz'

		else:
			raise NotImplementedError, "Invalid bagging model"

	# Train and predict
	print('Start training...')
	model.train(train_X, train_y, model_file)
	print('Finished training...')

	# Predictions
	print('Start predicting on training...')
	train_pred = model.predict(train_X, model_file)
	print('Finished predicting on training...')

	# Calculate CE/class rate on training set
	train_ce, train_class_rate = model.evaluate(train_y, train_pred, cross_entropy_flag = True)
	print("Training CE:\n{0}".format(train_ce))
	print("Training classification rate:\n{0}".format(train_class_rate))

	# Evaluate on valid set
	print('Start predicting on validation...')
	valid_pred = model.predict(valid_X, model_file)
	print('Finished predicting on validation...')

	# Calculate CE/class rate on validation set
	valid_ce, valid_class_rate = model.evaluate(valid_y, valid_pred, cross_entropy_flag = True)
	print("Validation CE:\n{0}".format(valid_ce))
	print("Validation classification rate:\n{0}".format(valid_class_rate))

	# Save predictions
	np.savez(train_output_file, predictions=train_pred)
	np.savez(valid_output_file, predictions=valid_pred)

	print("==================================")


# TODO Yuan - do we need this?

#def run_ada_logistic():

# #Test area
# print("Tests - knn")

# train_X = [[0], [1], [2], [3]]
# train_y = [0, 0, 1, 1]

# test_X = [[1.1], [0.9]]
# test_y = [1, 0]

# knn_alg = Model('knn', {'k':3})
# knn_alg.train(train_X, train_y, 'knn_test_model.npz')
# test_pred = knn_alg.predict(test_X, 'knn_test_model.npz')
# print(test_pred)

# ce, class_rate = knn_alg.evaluate(test_y, test_pred, cross_entropy_flag = True)
# print(ce)
# print(class_rate)


# print("Tests - logistic")

# logistic_alg = Model('logistic', {'penalty': 'l2', 'regularization_term':0.1})
# logistic_alg.train(train_X, train_y, 'logistic_test_model')
# test_pred = logistic_alg.predict(test_X, 'logistic_test_model')
# print(test_pred)

# ce, class_rate = logistic_alg.evaluate(test_y, test_pred, cross_entropy_flag = True)
# print(ce)
# print(class_rate)
