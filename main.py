#!/usr/bin/env python2.7
# -*- coding: utf8 -*-
from __future__ import absolute_import, print_function, unicode_literals
"""Main script for training and evaluating models"""

# Standard modules
import argparse, time, os, json, csv
# Third party modules
import numpy as np
import matplotlib.pyplot as plt
# Model implementations
from models import KNNModel, SVMModel, LogisticRegressionModel, NeuralNetworkModel, MOGModel, DecisionTreeModel, AdaboostModel, BaggingModel, RandomForestModel, ExtraTreesModel
from sklearn.ensemble import ExtraTreesClassifier

def create_dir_if_not_exists(filename):
	if not os.path.exists(os.path.dirname(filename)):
		os.makedirs(os.path.dirname(filename))

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
available_models = ['knn', 'logistic', 'svm', 'mog', 'neural_net', 'decision_tree', 'randomforest', 'extratrees']
available_ensemble_methods = ['adaboost', 'bagging']

# Set up command-line arguments
_argument_parser = argparse.ArgumentParser(
	prog='main.py',
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
_argument_parser.add_argument('-o', '--output',
	dest='output_file',
	help='the file to output results to',
	required=True
)
_argument_parser.add_argument('-p', '--params',
	dest='params',
	help='a JSON string containing parameters for the specified algorithm',
	required=True
)
_argument_parser.add_argument('--train-data',
	dest='train_data_file',
	help='the training data file to use',
	required=True
)
_argument_parser.add_argument('--train-predictions-output',
	dest='train_predictions_file',
	help='the file to write the training predictions to',
	required=True
)
_argument_parser.add_argument('--valid-data',
	dest='valid_data_file',
	help='the valid data file to use',
	required=True
)
_argument_parser.add_argument('--valid-predictions-output',
	dest='valid_predictions_file',
	help='the file to write the valid predictions to',
	required=True
)
_argument_parser.add_argument('--model-file',
	dest='model_file',
	help='the file to output the model to',
	required=True
)
_argument_parser.add_argument('--summary-file',
	dest='summary_file',
	help='the summary report file to output all metrics to',
	required=True
)
# TODO add arguments for file locations, etc.
args = _argument_parser.parse_args()

# Run script
if __name__ == '__main__':
	model_name = args.model_name
	ensemble_method = args.ensemble_method
	output_file = args.output_file
	params_json = args.params
	train_data_file = args.train_data_file
	train_output_file = args.train_predictions_file
	valid_data_file = args.valid_data_file
	valid_output_file = args.valid_predictions_file
	model_file = args.model_file
	summary_file = args.summary_file

	if not ensemble_method:
		print("Running {0} - trained on {1}".format(model_name, train_data_file))
	else:
		print("Running {0} ({1}) - trained on {2}".format(ensemble_method, model_name, train_data_file))

	#training set
	train_data = np.load(train_data_file)
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

	# Run classifier
	model = None
	params = json.loads(params_json)

	# Standard algorithms
	if not ensemble_method:
		# KNN
		if model_name == 'knn':
			model = KNNModel(params) # KNNModel({'k': 10})

		# Logistic
		elif model_name == 'logistic':
			model = LogisticRegressionModel(params) # LogisticRegressionModel({'penalty': 'l2', 'regularization_term': 0.1})

		# SVM
		elif model_name == 'svm':
			model = SVMModel(params) # SVMModel({'kernel': 'rbf', 'probability_flag': False})

		# MoG
		elif model_name == 'mog':
			model = MOGModel(params) # MOGModel({'n_components': 20})

		# Multi-layer perceptron (NNets)
		elif model_name == 'neural_net':
			model = NeuralNetworkModel(params) # NeuralNetworkModel({'xxx': 0})

		# Decision tree
		elif model_name == 'decision_tree':
			model = DecisionTreeModel(params) # DecisionTreeModel({'criterion': 'gini', 'max_depth': 10, 'min_samples_split': 5})

		#random forest
		elif (model_name == "randomforest"):
			model = RandomForestModel(params) # RandomForestModel({'n_estimators': 100, 'criterion': 'gini', 'max_depth': 3, 'min_samples_split': 5})

		#extra trees
		elif (model_name == "extratrees"):
			model = ExtraTreesModel(params) # ExtraTreesModel({'n_estimators': 100, 'criterion': 'gini', 'max_depth': 3, 'min_samples_split': 5})

		else:
			raise NotImplementedError, "Invalid model name"

	# Adaboost
	elif (ensemble_method == 'adaboost'):
		# Adaboost decision tree
		if model_name == 'decision_tree':
			model = AdaboostModel(params) # AdaboostModel({'algorithm_name': 'decision_tree', 'n_estimators': 25, 'criterion': 'gini', 'max_depth': 1, 'min_samples_split': 5})
			# model_file = 'ada_dt_test_model.npz'
			# train_output_file = 'examples-train-predictions-ada_dt-5000.npz'
			# valid_output_file = 'examples-valid-predictions-ada_dt-5000.npz'

		else:
			raise NotImplementedError, "Invalid adaboost model"

	# Bagging
	elif (ensemble_method == "bagging"):
		# Bagging decision tree
		if model_name == 'decision_tree':
			model = BaggingModel(params) # BaggingModel({'algorithm_name': 'decision_tree', 'n_estimators': 25, 'criterion': 'gini', 'max_depth': 3, 'min_samples_split': 5})
			# model_file = 'bag_dt_test_model.npz'
			# train_output_file = 'examples-train-predictions-bag_dt-5000.npz'
			# valid_output_file = 'examples-valid-predictions-bag_dt-5000.npz'

		elif model_name == 'logistic':
			model = BaggingModel(params) # BaggingModel({'algorithm_name': 'logistic', 'n_estimators': 25, 'penalty': 'l2', 'regularization_term': 0.01})
			# model_file = 'bag_log_test_model.npz'
			# train_output_file = 'examples-train-predictions-bag_log-5000.npz'
			# valid_output_file = 'examples-valid-predictions-bag_log-5000.npz'

		else:
			raise NotImplementedError, "Invalid bagging model"

	# Create any necessary directories if needed
	create_dir_if_not_exists(train_output_file)
	create_dir_if_not_exists(valid_output_file)
	create_dir_if_not_exists(model_file)
	create_dir_if_not_exists(output_file)

	# Train
	print('Start training...')
	start = time.time()
	model.train(train_X, train_y, model_file)
	end = time.time()
	train_time = end - start
	print('Finished training...')

	with open(output_file, 'w') as f:
		f.write("SUPERPIXEL LEVEL RESULTS\n\n")
		f.write("{0} - trained on {1}\n\n".format(model_name, train_data_file))
		f.write("Training time (s): \n\t{0}\n\n".format(train_time))

		# Predictions
		f.write("===================\n")
		f.write("TRAINING\n")
		f.write("===================\n")
		print('Start predicting on training...')
		start = time.time()
		train_pred = model.predict(train_X, model_file)
		end = time.time()
		train_time = end - start
		f.write("Training prediction time (s):\n\t{0}\n".format(train_time))
		print('Finished predicting on training...')

		# Calculate CE/class rate on training set
		train_ce, train_class_rate, train_precision, train_recall, train_f1_score = model.evaluate(train_y, train_pred, cross_entropy_flag = True)
		f.write("Training CE:\n\t{0}\n".format(train_ce))
		f.write("Training classification rate:\n\t{0}\n".format(train_class_rate))
		f.write("Training precision:\n\t{0}\n".format(train_precision))
		f.write("Training recall:\n\t{0}\n".format(train_recall))
		f.write("Training f1 score:\n\t{0}\n".format(train_f1_score))

		# Evaluate on valid set
		f.write("\n===================\n")
		f.write("VALID\n")
		f.write("===================\n")
		print('Start predicting on validation...')
		start = time.time()
		valid_pred = model.predict(valid_X, model_file)
		end = time.time()
		train_time = end - start
		f.write("Valid prediction time (s):\n\t{0}\n".format(train_time))
		print('Finished predicting on validation...')

		# Calculate CE/class rate on validation set
		valid_ce, valid_class_rate, valid_precision, valid_recall, valid_f1_score = model.evaluate(valid_y, valid_pred, cross_entropy_flag = True)
		f.write("Validation CE:\n\t{0}\n".format(valid_ce))
		f.write("Validation classification rate:\n\t{0}\n".format(valid_class_rate))
		f.write("Validation precision:\n\t{0}\n".format(valid_precision))
		f.write("Validation recall:\n\t{0}\n".format(valid_recall))
		f.write("Validation f1 score:\n\t{0}\n".format(valid_f1_score))

		# Also write to the summary file
		fieldnames = ['algorithm', 'configuration', 'data_set', 'ce', 'classification_rate', 'precision', 'recall', 'f1']
		
		if os.path.isfile(summary_file):
			report = open(summary_file, 'a')
			writer = csv.DictWriter(report, fieldnames=fieldnames)
		else:
			report = open(summary_file, 'w')
			writer = csv.DictWriter(report, fieldnames=fieldnames)
			writer.writeheader()
		
		# Write train results
		writer.writerow({
			'algorithm': model_name,
			'configuration': params_json,
			'data_set': 'train',
			'ce': train_ce,
			'classification_rate': train_class_rate,
			'precision': train_precision,
			'recall': train_recall,
			'f1': train_f1_score
		})
		# Write valid results
		writer.writerow({
			'algorithm': model_name,
			'configuration': params_json,
			'data_set': 'valid',
			'ce': valid_ce,
			'classification_rate': valid_class_rate,
			'precision': valid_precision,
			'recall': valid_recall,
			'f1': valid_f1_score
		})

		report.close()

	# Save predictions
	np.savez(train_output_file, predictions=train_pred)
	np.savez(valid_output_file, predictions=valid_pred)
