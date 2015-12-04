#!/usr/bin/env python2.7
# -*- coding: utf8 -*-
from __future__ import absolute_import, print_function, unicode_literals
"""defines classes that encapsulates the sci-kit learn ML algorithms"""

#general
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.externals import joblib

#model implementations
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neural_network
from sklearn import linear_model, decomposition, datasets #logistic regression
from sklearn import mixture
from sklearn.mixture import GMM
from sknn import mlp
import sknn
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier

#pipeline, cross validation, model selection
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

		elif (self.name == "mog_classifier"):
			return MoG_classifier_train(train_inputs, train_targets, self.hyperparameters_dic, model_name)

		elif (self.name == "mlp"):
			return mlp_train(train_inputs, train_targets, self.hyperparameters_dic, model_name)

		elif (self.name == "decision_tree"):
			return decision_tree_train(train_inputs, train_targets, self.hyperparameters_dic, model_name)

		elif (self.name == "adaboost"):
			return adaboost_train(train_inputs, train_targets, self.hyperparameters_dic, model_name)

		elif (self.name == "bagging"):
			return bagging_train(train_inputs, train_targets, self.hyperparameters_dic, model_name)

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

		elif (self.name == "mog_classifier"):
			return MoG_classifier_predict(test_inputs, self.hyperparameters_dic, model_name)

		elif (self.name == "mlp"):
			return mlp_predict(test_inputs, self.hyperparameters_dic, model_name)

		elif(self.name == "decision_tree"):
			return decision_tree_predict(test_inputs, self.hyperparameters_dic, model_name)

		elif (self.name == "adaboost"):
			return adaboost_predict(test_inputs, self.hyperparameters_dic, model_name)


		elif (self.name == "bagging"):
			return bagging_predict(test_inputs, self.hyperparameters_dic, model_name)

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
	k = hyperparameters_dic['k'] #value of k - integer

	#load model
	model_data = np.load(model_name)
	train_inputs = model_data['train_inputs']
	train_targets = model_data['train_targets'].ravel()

	#fit model
	neigh = KNeighborsClassifier(n_neighbors=k)
	neigh.fit(train_inputs, train_targets)

	#make predictions
	test_pred = np.array(neigh.predict_proba(test_inputs))
	print(test_pred.shape)
	return test_pred[:,1]


#SVM
def svm_train(train_inputs, train_targets, hyperparameters_dic, model_name):
	#force our train inputs and targets to be np arrays
	train_inputs = np.array(train_inputs)
	train_targets = np.array(train_targets).ravel()

	#unpack hyperparameters
	kernel = hyperparameters_dic['kernel'] #str: eg. 'rbf', 'poly', 'linear', 'sigmoid'
	probability_flag = hyperparameters_dic['probability_flag'] #True - returns a probability; False - return a hard assignment

	#fit model
	svm_mod = svm.SVC(probability = probability_flag)
	svm_mod.fit(train_inputs, train_targets)

	#save model
	joblib.dump(svm_mod, (model_name+'.pkl'))

def svm_predict(test_inputs, hyperparameters_dic, model_name):
	#force test inputs to be np arrays:
	test_inputs = np.array(test_inputs)

	#unpack hyperparameters
	kernel = hyperparameters_dic['kernel'] #eg. rbf
	probability_flag = hyperparameters_dic['probability_flag']

	#load model
	svm_mod = joblib.load((model_name+'.pkl'))

	#make predictions
	if (probability_flag == True):
	 	test_pred = np.array(svm_mod.predict_proba(test_inputs))[:, 1]
	else:
		test_pred = np.array(svm_mod.predict(test_inputs))

	return test_pred


#Logistic Regression
def logistic_train(train_inputs, train_targets, hyperparameters_dic, model_name):
	#force our train inputs and targets to be np arrays
	train_inputs = np.array(train_inputs)
	train_targets = np.array(train_targets).ravel()

	#unpack hyperparameters:
	penalty = hyperparameters_dic['penalty'] #str 'l1' or 'l2'
	regularization_term = hyperparameters_dic['regularization_term'] #float value

	#fit model
	logistic = linear_model.LogisticRegression(penalty= penalty, C=regularization_term)
	logistic.fit(train_inputs, train_targets)

	#save model
	joblib.dump(logistic, (model_name+'.pkl'))

def logistic_predict(test_inputs, hyperparameters_dic, model_name):
	#force test inputs to be np arrays:
	test_inputs = np.array(test_inputs)

	#load model
	logistic = joblib.load((model_name+'.pkl'))

	#make predictions
	test_pred = np.array(logistic.predict_proba(test_inputs))
	return test_pred[:, 1]

#Neural networks (aka multilayer perceptron - mlp)
def mlp_train(train_inputs, train_targets, hyperparameters_dic, model_name):
	#force our train inputs and targets to be np arrays
	train_inputs = np.array(train_inputs)
	train_targets = np.array(train_targets)

	#unpack hyperparameters:
	# hidden_layers_tuple = hyperparameters_dic['hidden_layers_tuple'] # tuple - (#hidden_units_layer_1)
	# activation = hyperparameters_dic['activation'] #str 'logistic', 'tanh', 'relu' => appearantly tanh and relu are better
	# optimization_alg = hyperparameters_dic['optimization_alg'] #str 'l-bfg' - all data or 'adam' - batch - default 200
	# L2_reg_alpha = hyperparameters_dic['L2_reg_alpha'] #L2 regularization term
	#learning_rate_type = hyperparameters_dic['learning_rate_type'] #'constant','invscaling' - decreasing, 'adaptive' - only used for sgd
	#random_state = hyperparameters_dic['random_state'] #int - initialization
	#learning_rate_init = hyperparameters_dic['learning_rate_init'] #double

	#fit model
	#temp code
	hidden_layer_1 = mlp.Layer(type = 'Tanh', units = 25, weight_decay = 0.01)
	#hidden_layer_2 = mlp.Layer(type = 'Tanh', units = 10, weight_decay = 0.01)
	#hidden_layer_3 = mlp.Layer(type = 'Tanh', units = 25, weight_decay = 0.01)
	output_layer = mlp.Layer(type = 'Softmax')

	list_of_layers = [hidden_layer_1, output_layer]


	mlp_model = mlp.Classifier(layers = list_of_layers)
	mlp_model.fit(train_inputs, train_targets)

	#save model
	joblib.dump(mlp_model, (model_name+'.pkl'))


def mlp_predict(test_inputs, hyperparameters_dic, model_name):
	#force test inputs to be np arrays:
	test_inputs = np.array(test_inputs)

	#load model
	mlp_model = joblib.load((model_name+'.pkl'))

	#make predictions
	test_pred = np.array(mlp_model.predict_proba(test_inputs))
	return test_pred[:, 1]

#Mixture of Gaussian Classifier
def MoG_classifier_train(train_inputs, train_targets, hyperparameters_dic, model_name):
	#force our train inputs and targets to be np arrays
	train_inputs = np.array(train_inputs)
	train_targets = np.array(train_targets).ravel()

	#unpack hyperparameters:
	n_components = hyperparameters_dic['n_components']


	#fit model
	MoG_class_0 = mixture.GMM(n_components = n_components)
	MoG_class_1 = mixture.GMM(n_components = n_components)

	train_targets_ravel=train_targets.ravel()

	MoG_class_0.fit(train_inputs[train_targets_ravel == 0])
	MoG_class_1.fit(train_inputs[train_targets_ravel == 1])

	prior_class_1 = np.array(sum(train_targets))/float(train_targets.shape[0])

	MoG_classifier = {'MoG_class_0': MoG_class_0, 'MoG_class_1': MoG_class_1, 'prior_class_1': prior_class_1}
	
	#save model
	joblib.dump(MoG_classifier, (model_name+'.pkl'))

def MoG_classifier_predict(test_inputs, hyperparameters_dic, model_name):
	#force test inputs to be np arrays:
	test_inputs = np.array(test_inputs)

	#unpack hyperparameters:
	n_components = hyperparameters_dic['n_components']

	#load model
	MoG_classifier = joblib.load((model_name+'.pkl'))
	MoG_class_0 = MoG_classifier['MoG_class_0']
	MoG_class_1 = MoG_classifier['MoG_class_1']
	prior_class_1 = MoG_classifier['prior_class_1']

	#make predictions
	class_0_logp = np.array(MoG_class_0.score(test_inputs))
	class_1_logp = np.array(MoG_class_1.score(test_inputs))


	p_x_cond_class0 = np.exp(class_0_logp)
	p_x_cond_class1 = np.exp(class_1_logp)

	p_class1_cond_x = np.divide((p_x_cond_class1)*prior_class_1, (p_x_cond_class1)*prior_class_1+(p_x_cond_class0)*(1-prior_class_1))
	p_class0_cond_x = np.divide((p_x_cond_class0)*(1-prior_class_1), (p_x_cond_class1)*prior_class_1+(p_x_cond_class0)*(1-prior_class_1))

	#assert(np.sum(p_class1_cond_x+p_class0_cond_x) = p_class1_cond_x.shape[0])

	return p_class1_cond_x

#Decision Tree:
def decision_tree_train(train_inputs, train_targets, hyperparameters_dic, model_name):
	#force our train inputs and targets to be np arrays
	train_inputs = np.array(train_inputs)
	train_targets = np.array(train_targets).ravel()

	#unpack hyperparamters
	criterion = hyperparameters_dic['criterion']
	max_depth = hyperparameters_dic['max_depth']
	min_samples_split = hyperparameters_dic['min_samples_split']

	#fit decision tree
	dt_classifier = tree.DecisionTreeClassifier(criterion = criterion, max_depth = max_depth, min_samples_split = min_samples_split)
	dt_classifier.fit(train_inputs, train_targets)

	#save model
	joblib.dump(dt_classifier, (model_name+'.pkl'))


def decision_tree_predict(test_inputs, hyperparameters_dic, model_name):
	#force test inputs to be np arrays:
	test_inputs = np.array(test_inputs)

	#load model
	dt_classifier = joblib.load((model_name+'.pkl'))

	#make predictions
	test_pred = dt_classifier.predict_proba(test_inputs)

	return test_pred[:, 1]

#Adaboost:
def adaboost_train(train_inputs, train_targets, hyperparameters_dic, model_name):
	#force our train inputs and targets to be np arrays
	train_inputs = np.array(train_inputs)
	train_targets = np.array(train_targets).ravel()

	#unpack hyperparamters
	algorithm_name = hyperparameters_dic['algorithm_name']
	n_estimators = hyperparameters_dic['n_estimators']

	if (algorithm_name == 'decision_tree'):
		base_classifier = decision_tree_init(hyperparameters_dic)

	elif (algorithm_name == 'logistic'):
		base_classifier = logistic_init(hyperparameters_dic)

	else:
		print('Error: Unexpected base classifier name.')

	#fit model
	adaboost_classifier = AdaBoostClassifier(base_classifier, n_estimators)
	adaboost_classifier.fit(train_inputs, train_targets)

	#save model
	joblib.dump(adaboost_classifier, (model_name+'.pkl'))

def adaboost_predict(test_inputs, hyperparameters_dic, model_name):
	#force test inputs to be np arrays:
	test_inputs = np.array(test_inputs)

	#load model
	adaboost_classifier = joblib.load((model_name+'.pkl'))

	#make predictions
	test_pred = adaboost_classifier.predict_proba(test_inputs)

	return test_pred[:, 1]

#Bagging:
def bagging_train(train_inputs, train_targets, hyperparameters_dic, model_name):
	#force our train inputs and targets to be np arrays
	train_inputs = np.array(train_inputs)
	train_targets = np.array(train_targets).ravel()

	#unpack hyperparamters
	algorithm_name = hyperparameters_dic['algorithm_name']
	n_estimators = hyperparameters_dic['n_estimators']

	if (algorithm_name == 'decision_tree'):
		base_classifier = decision_tree_init(hyperparameters_dic)

	elif (algorithm_name == 'logistic'):
		base_classifier = logistic_init(hyperparameters_dic)

	else:
		print('Error: Unexpected base classifier name.')

	#fit model
	bagging_classifier = BaggingClassifier(base_classifier, n_estimators)
	bagging_classifier.fit(train_inputs, train_targets)

	#save model
	joblib.dump(bagging_classifier, (model_name+'.pkl'))

def bagging_predict(test_inputs, hyperparameters_dic, model_name):
	#force test inputs to be np arrays:
	test_inputs = np.array(test_inputs)

	#load model
	bagging_classifier = joblib.load((model_name+'.pkl'))

	#make predictions
	test_pred = bagging_classifier.predict_proba(test_inputs)

	return test_pred[:, 1]


#ensemble method base classifier initialization functions
def decision_tree_init(hyperparameters_dic):
	#unpack hyperparamters
	criterion = hyperparameters_dic['criterion']
	max_depth = hyperparameters_dic['max_depth']
	min_samples_split = hyperparameters_dic['min_samples_split']

	#fit decision tree
	dt_classifier = tree.DecisionTreeClassifier(criterion = criterion, max_depth = max_depth, min_samples_split = min_samples_split)

	return dt_classifier

def logistic_init(hyperparameters_dic):
	#unpack hyperparameters:
	penalty = hyperparameters_dic['penalty'] #str 'l1' or 'l2'
	regularization_term = hyperparameters_dic['regularization_term'] #float value

	#fit model
	logistic_classifier = linear_model.LogisticRegression(penalty= penalty, C=regularization_term)


	return logistic_classifier








#========
#Testing
#========

#training set
train_data = np.load('train_examples_5000sp.npz')
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
valid_data = np.load('valid_examples_5000sp.npz')
valid_X = valid_data['inputs']
valid_y = valid_data['targets']

#normalize the value of features to be between -1 to 1
for j in xrange(valid_X.shape[1]):
	valid_X[:,j] = (valid_X[:,j] - train_X_features_mean[0,j])/train_X_features_range[0,j]


#test data
test_data = np.load('test_examples_5000sp.npz')
test_X = valid_data['inputs']
test_y = valid_data['targets']

#normalize the value of features to be between -1 to 1
for j in xrange(test_X.shape[1]):
	test_X[:,j] = (test_X[:,j] - train_X_features_mean[0,j])/train_X_features_range[0,j]



def run_feature_importance():
	#feature importance forest
	# Build a forest and compute the feature importances
	print('Feature Importance using random forest')
	forest = ExtraTreesClassifier(n_estimators=250,
	                              random_state=0)

	forest.fit(train_X, train_y)
	importances = forest.feature_importances_
	std = np.std([tree.feature_importances_ for tree in forest.estimators_],
	             axis=0)
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


#run classifiers

print('==================================')
print('Running Classifiers...')
print('==================================')

def run_k_nn():
	print('running KNN...')

	knn_alg = ScikitLearnML('knn', {'k':10})
	knn_alg.train(train_X, train_y, 'knn_test_model.npz')

	#evaluate on training set
	train_pred = knn_alg.predict(train_X, 'knn_test_model.npz')

	train_ce, train_class_rate = knn_alg.evaluate(train_y, train_pred, cross_entropy_flag = True)
	print("training CE:")
	print(train_ce)
	print("training classification rate:")
	print(train_class_rate)

	#evaluate on valid set
	valid_pred = knn_alg.predict(valid_X, 'knn_test_model.npz')

	valid_ce, valid_class_rate = knn_alg.evaluate(valid_y, valid_pred, cross_entropy_flag = True)
	print("validation CE:")
	print(valid_ce)
	print("validation classification rate:")
	print(valid_class_rate)


	np.savez('examples-train-predictions-knn-5000.npz', predictions=train_pred)
	np.savez('examples-valid-predictions-knn-5000.npz', predictions=valid_pred)

	print('==================================')


#def run_logistic():
print('running logistic...')

logistic_alg = ScikitLearnML('logistic', {'penalty': 'l2', 'regularization_term':0.1})
logistic_alg.train(train_X, train_y, 'logistic_test_model.npz')

#evaluate on training set
train_pred = logistic_alg.predict(train_X, 'logistic_test_model.npz')

train_ce, train_class_rate = logistic_alg.evaluate(train_y, train_pred, cross_entropy_flag = True)
print("training CE:")
print(train_ce)
print("training classification rate:")
print(train_class_rate)

#evaluate on valid set
valid_pred = logistic_alg.predict(valid_X, 'logistic_test_model.npz')

valid_ce, valid_class_rate = logistic_alg.evaluate(valid_y, valid_pred, cross_entropy_flag = True)
print("validation CE:")
print(valid_ce)
print("validation classification rate:")
print(valid_class_rate)


np.savez('examples-train-predictions-logistic-5000.npz', predictions=train_pred)
np.savez('examples-valid-predictions-logistic-5000.npz', predictions=valid_pred)


print('==================================')

def run_nn():
	print('running neural net classifier...')

	mlp_alg = ScikitLearnML('mlp', {'xxx': 0})
	mlp_alg.train(train_X, train_y, 'mlp_test_model.npz')

	print('finished training...')

	#evaluate on training set
	train_pred = mlp_alg.predict(train_X, 'mlp_test_model.npz')

	train_ce, train_class_rate = mlp_alg.evaluate(train_y, train_pred, cross_entropy_flag = True)
	print("training CE:")
	print(train_ce)
	print("training classification rate:")
	print(train_class_rate)

	#evaluate on valid set
	valid_pred = mlp_alg.predict(valid_X, 'mlp_test_model.npz')

	valid_ce, valid_class_rate = mlp_alg.evaluate(valid_y, valid_pred, cross_entropy_flag = True)
	print("validation CE:")
	print(valid_ce)
	print("validation classification rate:")
	print(valid_class_rate)


	np.savez('examples-train-predictions-mlp.npz', predictions=train_pred)
	np.savez('examples-valid-predictions-mlp.npz', predictions=valid_pred)


	print('==================================')


def run_mog():
	print('running MoG_classifier...')

	mog_alg = ScikitLearnML('mog_classifier', {'n_components': 20})
	mog_alg.train(train_X, train_y, 'mog_test_model.npz')

	print('finished training...')

	#evaluate on training set
	train_pred = mog_alg.predict(train_X, 'mog_test_model.npz')

	train_ce, train_class_rate = mog_alg.evaluate(train_y, train_pred, cross_entropy_flag = True)
	print("training CE:")
	print(train_ce)
	print("training classification rate:")
	print(train_class_rate)

	#evaluate on valid set
	valid_pred = mog_alg.predict(valid_X, 'mog_test_model.npz')

	valid_ce, valid_class_rate = mog_alg.evaluate(valid_y, valid_pred, cross_entropy_flag = True)
	print("validation CE:")
	print(valid_ce)
	print("validation classification rate:")
	print(valid_class_rate)


	np.savez('examples-train-predictions-mog-5000.npz', predictions=train_pred)
	np.savez('examples-valid-predictions-mog-5000.npz', predictions=valid_pred)

	print('==================================')

def run_svm():
	print('running svm...')

	svm_alg = ScikitLearnML('svm', {'kernel': "rbf", 'probability_flag': False})
	svm_alg.train(train_X, train_y, 'svm_test_model.npz')
	print('finished training...')

	svm_alg = ScikitLearnML('svm', {'kernel': "rbf", 'probability_flag': False})

	#evaluate on training set
	print('start predicting...')
	train_pred = svm_alg.predict(train_X, 'svm_test_model.npz')
	print('finished predicting on training...')

	train_ce, train_class_rate = svm_alg.evaluate(train_y, train_pred, cross_entropy_flag = True)
	print("training CE:")
	print(train_ce)
	print("training classification rate:")
	print(train_class_rate)

	#evaluate on valid set
	valid_pred = svm_alg.predict(valid_X, 'svm_test_model.npz')
	print('finished predicting on validation...')

	valid_ce, valid_class_rate = svm_alg.evaluate(valid_y, valid_pred, cross_entropy_flag = True)
	print("validation CE:")
	print(valid_ce)
	print("validation classification rate:")
	print(valid_class_rate)


	np.savez('examples-train-predictions-svm.npz', predictions=train_pred)
	np.savez('examples-valid-predictions-svm.npz', predictions=valid_pred)

	print('==================================')


#def run_decision_tree():

print('running decision tree...')

dt_alg = ScikitLearnML('decision_tree', {'criterion':'gini', 'max_depth':10, 'min_samples_split': 5})
dt_alg.train(train_X, train_y, 'decision_tree_test_model.npz')

#evaluate on training set
train_pred = dt_alg.predict(train_X, 'decision_tree_test_model.npz')

train_ce, train_class_rate = dt_alg.evaluate(train_y, train_pred, cross_entropy_flag = True)
print("training CE:")
print(train_ce)
print("training classification rate:")
print(train_class_rate)

#evaluate on valid set
valid_pred = dt_alg.predict(valid_X, 'decision_tree_test_model.npz')

valid_ce, valid_class_rate = dt_alg.evaluate(valid_y, valid_pred, cross_entropy_flag = True)
print("validation CE:")
print(valid_ce)
print("validation classification rate:")
print(valid_class_rate)


np.savez('examples-train-predictions-decisiontree-5000.npz', predictions=train_pred)
np.savez('examples-valid-predictions-decisiontree-5000.npz', predictions=valid_pred)

print('==========================================')

def run_ada_decision_stump():
	print('running adaboost - decision stump')

	ada_dt_alg = ScikitLearnML('adaboost', {'algorithm_name': 'decision_tree', 'n_estimators': 25, 'criterion':'gini', 'max_depth':1, 'min_samples_split': 5})
	ada_dt_alg.train(train_X, train_y, 'ada_dt_test_model.npz')

	#evaluate on training set
	train_pred = ada_dt_alg.predict(train_X, 'ada_dt_test_model.npz')

	train_ce, train_class_rate = ada_dt_alg.evaluate(train_y, train_pred, cross_entropy_flag = True)
	print("training CE:")
	print(train_ce)
	print("training classification rate:")
	print(train_class_rate)

	#evaluate on valid set
	valid_pred = ada_dt_alg.predict(valid_X, 'ada_dt_test_model.npz')

	valid_ce, valid_class_rate = ada_dt_alg.evaluate(valid_y, valid_pred, cross_entropy_flag = True)
	print("validation CE:")
	print(valid_ce)
	print("validation classification rate:")
	print(valid_class_rate)


	np.savez('examples-train-predictions-ada_dt-5000.npz', predictions=train_pred)
	np.savez('examples-valid-predictions-ada_dt-5000.npz', predictions=valid_pred)

	print('==========================================')


def run_ada_decision_tree():
	print('running adaboost - decision tree')

	ada_dt_alg = ScikitLearnML('adaboost', {'algorithm_name': 'decision_tree', 'n_estimators': 25, 'criterion':'gini', 'max_depth':3, 'min_samples_split': 5})
	ada_dt_alg.train(train_X, train_y, 'ada_dt_test_model.npz')

	#evaluate on training set
	train_pred = ada_dt_alg.predict(train_X, 'ada_dt_test_model.npz')

	train_ce, train_class_rate = ada_dt_alg.evaluate(train_y, train_pred, cross_entropy_flag = True)
	print("training CE:")
	print(train_ce)
	print("training classification rate:")
	print(train_class_rate)

	#evaluate on valid set
	valid_pred = ada_dt_alg.predict(valid_X, 'ada_dt_test_model.npz')

	valid_ce, valid_class_rate = ada_dt_alg.evaluate(valid_y, valid_pred, cross_entropy_flag = True)
	print("validation CE:")
	print(valid_ce)
	print("validation classification rate:")
	print(valid_class_rate)


	np.savez('examples-train-predictions-ada_dt-5000.npz', predictions=train_pred)
	np.savez('examples-valid-predictions-ada_dt-5000.npz', predictions=valid_pred)


def run_bagging_decision_tree():
	print('running bagging - decision tree')

	bag_dt_alg = ScikitLearnML('bagging', {'algorithm_name': 'decision_tree', 'n_estimators': 25, 'criterion':'gini', 'max_depth':3, 'min_samples_split': 5})
	bag_dt_alg.train(train_X, train_y, 'bag_dt_test_model.npz')

	#evaluate on training set
	train_pred = bag_dt_alg.predict(train_X, 'bag_dt_test_model.npz')

	train_ce, train_class_rate = bag_dt_alg.evaluate(train_y, train_pred, cross_entropy_flag = True)
	print("training CE:")
	print(train_ce)
	print("training classification rate:")
	print(train_class_rate)

	#evaluate on valid set
	valid_pred = bag_dt_alg.predict(valid_X, 'bag_dt_test_model.npz')

	valid_ce, valid_class_rate = bag_dt_alg.evaluate(valid_y, valid_pred, cross_entropy_flag = True)
	print("validation CE:")
	print(valid_ce)
	print("validation classification rate:")
	print(valid_class_rate)


	np.savez('examples-train-predictions-bag_dt-5000.npz', predictions=train_pred)
	np.savez('examples-valid-predictions-bag_dt-5000.npz', predictions=valid_pred)

	print("==================================")

#def run_bagging_logistic():
print('running bagging - logistic')

bag_log_alg = ScikitLearnML('bagging', {'algorithm_name': 'logistic', 'n_estimators': 25, 'penalty': 'l2', 'regularization_term': 0.01})
bag_log_alg.train(train_X, train_y, 'bag_log_test_model.npz')

#evaluate on training set
train_pred = bag_log_alg.predict(train_X, 'bag_log_test_model.npz')

train_ce, train_class_rate = bag_log_alg.evaluate(train_y, train_pred, cross_entropy_flag = True)
print("training CE:")
print(train_ce)
print("training classification rate:")
print(train_class_rate)

#evaluate on valid set
valid_pred = bag_log_alg.predict(valid_X, 'bag_log_test_model.npz')

valid_ce, valid_class_rate = bag_log_alg.evaluate(valid_y, valid_pred, cross_entropy_flag = True)
print("validation CE:")
print(valid_ce)
print("validation classification rate:")
print(valid_class_rate)


np.savez('examples-train-predictions-bag_log-5000.npz', predictions=train_pred)
np.savez('examples-valid-predictions-bag_log-5000.npz', predictions=valid_pred)

print("===================================")

#def run_ada_logistic():

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

