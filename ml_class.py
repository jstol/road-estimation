#!/usr/bin/env python2.7
# -*- coding: utf8 -*-
from __future__ import absolute_import, print_function, unicode_literals
"""defines classes that encapsulates the sci-kit learn ML algorithms"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.externals import joblib

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV



#ScikitLearnML Class:
class ScikitLearnML:
	def __init__(self, name, hyperparameters_dic):
		"""
		Attributes:
			- name:				string identifying the ML algorithm (eg. knn)
			- hyperparameters_dic:	hyperparameters stored as a dictionary (eg. {'k': 3})

		"""
		self.name = name
		self.hyperparameters_dic = hyperparameters_dic

	def train(self, train_inputs, train_targets, model_name):
		"""
		Inputs:
			- train inputs:		N x M matrix of input features (training set)
			- train targets:	1 x N array of target outputs (training set)
			- model name:		string identifying where to save the model

		Outputs:
			- None
		"""
		if (self.name == "knn"):
			return knn_train(train_inputs, train_targets, self.hyperparameters_dic, model_name)

		elif (self.name == "logistic"):
			return logistic_train(train_inputs, train_targets, self.hyperparameters_dic, model_name)

		elif (self.name == "svm"):
			return svm_train(train_inputs, train_targets, self.hyperparameters_dic, model_name)

		else:
			print("Error: unexpected ML algorithm name")

	def predict(self, test_inputs, model_name):
		"""
		Inputs:
			- test_inputs:		N x M matrix of input features (test set)
			- model name:		string identifying where to load the model

		Outputs:
			- test_pred:		1 x N array of predicted target (test set)
		"""

		if (self.name == "knn"):
			return knn_predict(test_inputs, self.hyperparameters_dic, model_name)

		elif (self.name == "logistic"):
			return logistic_predict(test_inputs, self.hyperparameters_dic, model_name)

		elif (self.name == "svm"):
			return svm_predict(test_inputs, self.hyperparameters_dic, model_name)

		else:
			print("Error: unexpected ML algorithm name")
			return None

	def evaluate(self, targets, predictions, cross_entropy_flag=False):
		"""
		Inputs:
			- targets:		1 x N array of target outputs
			- predicitions:	1 x N array of predicted target outputs

		Outputs:
			- class_rate:	scalar rate of correct responses
			- ce:			scalar cross entropy if the the cross entropy flag is true (appropriate if the test predictions are probabilities)
		"""
		targets = np.array(targets).ravel()
		predictions = np.array(predictions).ravel()

		# Cross entropy
		if (cross_entropy_flag == True):
			predictions_t1 = 0.5*np.ones(predictions.shape)
			predictions_t0 = 0.5*np.ones(predictions.shape)
			predictions_t1[targets == 1] = predictions[targets == 1]
			predictions_t0[targets == 0] = predictions[targets == 0]
			ce = -(np.dot(targets, np.log(predictions_t1)) + np.dot((1-targets), np.log(1-predictions_t0)))/float(len(targets))
		else:
			ce = None

		# Calculate the fraction of correct predictions (with y_i < 0.5 being class 0 and y_i >= 0.5 being class 1)
		predicted_class = np.round(predictions)
		class_rate = np.sum(np.equal(targets, predicted_class))/float(len(targets))

		return ce, class_rate




#K Nearest Neighbours
def knn_train(train_inputs, train_targets, hyperparameters_dic, model_name):
	train_inputs = np.array(train_inputs)
	train_targets = np.array(train_targets)
	np.savez(model_name, train_inputs = train_inputs, train_targets = train_targets)
	return None

def knn_predict(test_inputs, hyperparameters_dic, model_name):
	#force test inputs to be np arrays:
	test_inputs = np.array(test_inputs)

	#unpack hyperparameters
	k = hyperparameters_dic['k']

	#load model
	model_data = np.load(model_name)
	train_inputs = model_data['train_inputs']
	train_targets = model_data['train_targets']

	#fit model
	neigh = KNeighborsClassifier(n_neighbors=k)
	neigh.fit(train_inputs, train_targets)

	#make predictions
	test_pred = np.array(neigh.predict_proba(test_inputs))
	return test_pred[:, 1]


#SVM
def svm_train(train_inputs, train_targets, hyperparameters_dic, model_name):
	#force our train inputs and targets to be np arrays
	train_inputs = np.array(train_inputs)
	train_targets = np.array(train_targets)

	#unpack hyperparameters
	kernel = hyperparameters_dic['kernel']
	probability_flag = hyperparameters_dic['probability_flag']

	#fit model
	svm_mod = svm.SVC(kernel = kernel, probability = probability_flag)
	svm_mod.fit(train_inputs, train_targets)

	#save model
	joblib.dump(svm_mod, (model_name+'.pkl'))

def svm_predict(test_inputs, hyperparameters_dic, model_name):
	#force test inputs to be np arrays:
	test_inputs = np.array(test_inputs)

	#unpack hyperparameters
	kernel = hyperparameters_dic['kernel']
	probability_flag = hyperparameters_dic['probability_flag']

	#load model
	svm_mod = joblib.load((model_name+'.pkl'))

	#make predictions
	test_pred = np.array(svm_mod.predict_proba(test_inputs))
	return test_pred[:, 1]


#Logistic Regression
def logistic_train(train_inputs, train_targets, hyperparameters_dic, model_name):
	#force our train inputs and targets to be np arrays
	train_inputs = np.array(train_inputs)
	train_targets = np.array(train_targets)

	#unpack hyperparameters:
	penalty = hyperparameters_dic['penalty'] #str 'l1' or 'l2'
	regularization_term = hyperparameters_dic['regularization_term'] #value

	#fit model
	logistic = linear_model.LogisticRegression(penalty= penalty, C=regularization_term)
	logistic.fit(train_inputs, train_targets)

	#save model
	joblib.dump(logistic, (model_name+'.pkl'))

def logistic_predict(test_inputs, hyperparameters_dic, model_name):
	#force test inputs to be np arrays:
	test_inputs = np.array(test_inputs)

	#unpack hyperparameters:
	penalty = hyperparameters_dic['penalty'] #str 'l1' or 'l2'
	regularization_term = hyperparameters_dic['regularization_term'] #value

	#load model
	logistic = joblib.load((model_name+'.pkl'))

	#make predictions
	test_pred = np.array(logistic.predict_proba(test_inputs))
	return test_pred[:, 1]



train_data = np.load('examples.npz')
train_X = train_data['inputs']
train_y = train_data['targets']

knn_alg = ScikitLearnML('knn', {'k':3})
knn_alg.train(train_X, train_y, 'knn_test_model.npz')
train_pred = knn_alg.predict(train_X, 'knn_test_model.npz')
print(test_pred)

ce, class_rate = knn_alg.evaluate(train_y, train_pred, cross_entropy_flag = True)
print(ce)
print(class_rate)

np.savez('examples-predictions.npz', predictions=train_pred)



# #Test area
# print("Tests - knn")

# train_X = [[0], [1], [2], [3]]
# train_y = [0, 0, 1, 1]

# test_X = [[1.1], [0.9]]
# test_y = [1, 0]

# knn_alg = ScikitLearnML('knn', {'k':3})
# knn_alg.train(train_X, train_y, 'knn_test_model.npz')
# test_pred = knn_alg.predict(test_X, 'knn_test_model.npz')
# print(test_pred)

# ce, class_rate = knn_alg.evaluate(test_y, test_pred, cross_entropy_flag = True)
# print(ce)
# print(class_rate)


# print("Tests - logistic")

# logistic_alg = ScikitLearnML('logistic', {'penalty': 'l2', 'regularization_term':0.1})
# logistic_alg.train(train_X, train_y, 'logistic_test_model')
# test_pred = logistic_alg.predict(test_X, 'logistic_test_model')
# print(test_pred)

# ce, class_rate = logistic_alg.evaluate(test_y, test_pred, cross_entropy_flag = True)
# print(ce)
# print(class_rate)

