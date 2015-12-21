import os, cv2, time, csv, fnmatch
from glob import glob
import numpy as np
from datetime import datetime
from skimage.io import imsave


#general
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.externals import joblib
import math
import random

#image processing
from skimage.io import imread_collection, imread, imshow, imsave
from skimage import img_as_float, img_as_uint
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

#markov random field
from markovrandomfield_bishop import pixelmap



# STUFF NEEDED FROM THE DEV KIT -----------------------------------------------------------
# Some class they make use of
class dataStructure: 
	'''
	All the defines go in here!
	'''
	cats = ['um_road', 'umm_road', 'uu_road']
	calib_end = '.txt'
	im_end = '.png'
	gt_end = '.png'
	prob_end = '.png'
	eval_propertyList = ['MaxF', 'AvgPrec', 'PRE_wp', 'REC_wp', 'FPR_wp', 'FNR_wp' ] 

# COPIED FROM helper.py
def getGroundTruth(fileNameGT):
    '''
    Returns the ground truth maps for roadArea and the validArea 
    :param fileNameGT:
    '''
    # Read GT
    assert os.path.isfile(fileNameGT), 'Cannot find: %s' % fileNameGT
    full_gt = cv2.imread(fileNameGT, cv2.CV_LOAD_IMAGE_UNCHANGED if hasattr(cv2, 'CV_LOAD_IMAGE_UNCHANGED') else cv2.IMREAD_UNCHANGED)
    #attention: OpenCV reads in as BGR, so first channel has Blue / road GT
    roadArea =  full_gt[:,:,0] > 0
    validArea = full_gt[:,:,2] > 0

    return roadArea, validArea

# COPIED FROM helper.py
def calcEvalMeasures(evalDict, tag  = '_wp'):
    '''
    
    :param evalDict:
    :param tag:
    '''
    # array mode!
    TP = evalDict[:,0].astype('f4')
    TN = evalDict[:,1].astype('f4')
    FP = evalDict[:,2].astype('f4')
    FN = evalDict[:,3].astype('f4')
    Q = TP / (TP + FP + FN)
    P = TP + FN
    N = TN + FP
    TPR = TP / P
    FPR = FP / N
    FNR = FN / P
    TNR = TN / N
    A = (TP + TN) / (P + N)
    precision = TP / (TP + FP)
    recall = TP / P
    #numSamples = TP + TN + FP + FN
    correct_rate = A

    # F-measure
    #beta = 1.0
    #betasq = beta**2
    #F_max = (1 + betasq) * (precision * recall)/((betasq * precision) + recall + 1e-10)
    
    
    outDict =dict()

    outDict['TP'+ tag] = TP
    outDict['FP'+ tag] = FP
    outDict['FN'+ tag] = FN
    outDict['TN'+ tag] = TN
    outDict['Q'+ tag] = Q
    outDict['A'+ tag] = A
    outDict['TPR'+ tag] = TPR
    outDict['FPR'+ tag] = FPR
    outDict['FNR'+ tag] = FNR
    outDict['PRE'+ tag] = precision
    outDict['REC'+ tag] = recall
    outDict['correct_rate'+ tag] = correct_rate
    return outDict

# COPIED FROM helper.py
def evalExp(gtBin, cur_prob, thres, validMap = None, validArea=None):
	'''
	Does the basic pixel based evaluation!
	:param gtBin:
	:param cur_prob:
	:param thres:
	:param validMap:
	'''

	assert len(cur_prob.shape) == 2, 'Wrong size of input prob map'
	assert len(gtBin.shape) == 2, 'Wrong size of input prob map'
	thresInf = np.concatenate(([-np.Inf], thres, [np.Inf]))
	
	#Merge validMap with validArea
	if validMap is not None:
		if validArea is not None:
			validMap = (validMap == True) & (validArea == True)
	elif validArea is not None:
		validMap=validArea

	# histogram of false negatives
	if validMap is not None:
		fnArray = cur_prob[(gtBin == True) & (validMap == True)]
	else:
		fnArray = cur_prob[(gtBin == True)]
	fnHist = np.histogram(fnArray,bins=thresInf)[0]
	fnCum = np.cumsum(fnHist)
	FN = fnCum[0:0+len(thres)];
	
	if validMap is not None:
		fpArray = cur_prob[(gtBin == False) & (validMap == True)]
	else:
		fpArray = cur_prob[(gtBin == False)]
	
	fpHist  = np.histogram(fpArray, bins=thresInf)[0]
	fpCum = np.flipud(np.cumsum(np.flipud(fpHist)))
	FP = fpCum[1:1+len(thres)]

	# count labels and protos
	#posNum = fnArray.shape[0]
	#negNum = fpArray.shape[0]
	if validMap is not None:
		posNum = np.sum((gtBin == True) & (validMap == True))
		negNum = np.sum((gtBin == False) & (validMap == True))
	else:
		posNum = np.sum(gtBin == True)
		negNum = np.sum(gtBin == False)
	return FN, FP, posNum, negNum

# COPIED FROM helper.py
def pxEval_maximizeFMeasure(totalPosNum, totalNegNum, totalFN, totalFP, thresh = None):
	'''

	@param totalPosNum: scalar
	@param totalNegNum: scalar
	@param totalFN: vector
	@param totalFP: vector
	@param thresh: vector
	'''

	#Calc missing stuff
	totalTP = totalPosNum - totalFN
	totalTN = totalNegNum - totalFP


	valid = (totalTP>=0) & (totalTN>=0)
	assert valid.all(), 'Detected invalid elements in eval'

	recall = totalTP / float( totalPosNum )
	precision =  totalTP / (totalTP + totalFP + 1e-10)
	
	selector_invalid = (recall==0) & (precision==0)
	recall = recall[~selector_invalid]
	precision = precision[~selector_invalid]
		
	maxValidIndex = len(precision)
	
	#Pascal VOC average precision
	AvgPrec = 0
	counter = 0
	for i in np.arange(0,1.1,0.1):
		ind = np.where(recall>=i)
		if ind is None:
			continue
		pmax = max(precision[ind])
		AvgPrec += pmax
		counter += 1
	AvgPrec = AvgPrec/counter
	
	
	# F-measure operation point
	beta = 1.0
	betasq = beta**2
	F = (1 + betasq) * (precision * recall)/((betasq * precision) + recall + 1e-10)
	index = F.argmax()
	MaxF= F[index]
	
	recall_bst = recall[index]
	precision_bst =  precision[index]

	TP = totalTP[index]
	TN = totalTN[index]
	FP = totalFP[index]
	FN = totalFN[index]
	valuesMaxF = np.zeros((1,4),'u4')
	valuesMaxF[0,0] = TP
	valuesMaxF[0,1] = TN
	valuesMaxF[0,2] = FP
	valuesMaxF[0,3] = FN

	#ACC = (totalTP+ totalTN)/(totalPosNum+totalNegNum)
	prob_eval_scores  = calcEvalMeasures(valuesMaxF)
	prob_eval_scores['AvgPrec'] = AvgPrec
	prob_eval_scores['MaxF'] = MaxF

	#prob_eval_scores['totalFN'] = totalFN
	#prob_eval_scores['totalFP'] = totalFP
	prob_eval_scores['totalPosNum'] = totalPosNum
	prob_eval_scores['totalNegNum'] = totalNegNum

	prob_eval_scores['precision'] = precision
	prob_eval_scores['recall'] = recall
	#prob_eval_scores['precision_bst'] = precision_bst
	#prob_eval_scores['recall_bst'] = recall_bst
	prob_eval_scores['thresh'] = thresh
	if thresh is not None:
		BestThresh= thresh[index]
		prob_eval_scores['BestThresh'] = BestThresh

	#return a dict
	return prob_eval_scores

# evaluateRoad.oy "main"
def evaluate(result_dir, train_dir, summary_file, model_name, config_string, data_set, superpixels, debug = False):
    '''
    main method of evaluateRoad
    :param result_dir: directory with the result propability maps, e.g., /home/elvis/kitti_road/my_results
    :param gt_dir: training directory (has to contain gt_image_2)  e.g., /home/elvis/kitti_road/training
    :param debug: debug flag (OPTIONAL)
    '''
    
    # print "Starting evaluation ..." 
    # print "Available categories are: %s" %dataStructure.cats
    
    thresh = np.array(range(0,256))/255.0
    trainData_subdir_gt = 'gt_image_2/'
    gt_dir = os.path.join(train_dir,trainData_subdir_gt)
    
    assert os.path.isdir(result_dir), 'Cannot find result_dir: %s ' %result_dir
    
    # In the submission_dir we expect the probmaps! 
    submission_dir = result_dir
    assert os.path.isdir(submission_dir), 'Cannot find %s, ' %submission_dir
    
    # init result
    prob_eval_scores = [] # the eval results in a dict
    eval_cats = [] # saves al categories at were evaluated
    outputline = []
    for cat in dataStructure.cats:
        print "\nExecute evaluation for category %s ..." %cat
        fn_search  = '%s*%s' %(cat, dataStructure.gt_end)
        gt_fileList = glob(os.path.join(gt_dir, fn_search))
        assert len(gt_fileList)>0, 'Error reading ground truth'
        # Init data for categgory
        category_ok = True # Flag for each cat
        totalFP = np.zeros( thresh.shape )
        totalFN = np.zeros( thresh.shape )
        totalPosNum = 0
        totalNegNum = 0
        
        firstFile  = gt_fileList[0]
        file_key = firstFile.split('/')[-1].split('.')[0]
        tags = file_key.split('_')
        ts_tag = tags[2]
        dataset_tag = tags[0]
        class_tag = tags[1]
        
        submission_tag = dataset_tag + '_' + class_tag + '_'
        # print "Searching for submitted files with prefix: %s" %submission_tag
        
        for fn_curGt in gt_fileList:
            
            file_key = fn_curGt.split('/')[-1].split('.')[0]
            if debug:
                print "Processing file: %s " %file_key
            
            # get tags
            tags = file_key.split('_')
            ts_tag = tags[2]
            dataset_tag = tags[0]
            class_tag = tags[1]
            

            # Read GT
            cur_gt, validArea = getGroundTruth(fn_curGt)
                        
            # Read probmap and normalize
            fn_curProb = os.path.join(submission_dir, file_key + dataStructure.prob_end)
            
            if not os.path.isfile(fn_curProb):
                print "Cannot find file: %s for category %s." %(file_key, cat)
                print "--> Will now abort evaluation for this particular category."
                category_ok = False
                break
            
            cur_prob = cv2.imread(fn_curProb,0)
            cur_prob = np.clip( (cur_prob.astype('f4'))/(np.iinfo(cur_prob.dtype).max),0.,1.)
            
            FN, FP, posNum, negNum = evalExp(cur_gt, cur_prob, thresh, validMap = None, validArea=validArea)
            
            assert FN.max()<=posNum, 'BUG @ poitive samples'
            assert FP.max()<=negNum, 'BUG @ negative samples'
            
            # collect results for whole category
            totalFP += FP
            totalFN += FN
            totalPosNum += posNum
            totalNegNum += negNum
        
        if category_ok:
            # print "Computing evaluation scores..."
            # Compute eval scores!
            metrics = pxEval_maximizeFMeasure(totalPosNum, totalNegNum, totalFN, totalFP, thresh = thresh)
            prob_eval_scores.append(metrics)
            eval_cats.append(cat)

            metrics_dict = {}
            factor = 100
            for prop in dataStructure.eval_propertyList:
                metrics_dict[prop] = prob_eval_scores[-1][prop]*factor
                print '%s:\n\t%4.2f ' %(prop, metrics_dict[prop],)
            
            # Write to summary file
            if not os.path.exists(os.path.dirname(summary_file)):
				os.makedirs(os.path.dirname(summary_file))

            fieldnames = ['algorithm', 'configuration', 'superpixels', 'data_set', 'category'] + dataStructure.eval_propertyList
            if os.path.isfile(summary_file):
                report = open(summary_file, 'a')
                writer = csv.DictWriter(report, fieldnames=fieldnames)
            else:
                report = open(summary_file, 'w')
                writer = csv.DictWriter(report, fieldnames=fieldnames)
                writer.writeheader()
            
            # Reformat metrics
            for key in metrics_dict:
                metrics_dict[key] = float(metrics_dict[key])

            metrics_dict['algorithm'] = model_name
            metrics_dict['configuration'] = config_string
            metrics_dict['superpixels'] = superpixels
            metrics_dict['data_set'] = data_set
            metrics_dict['category'] = cat

            writer.writerow(metrics_dict)
            report.close()

            # print "Finished evaluating category: %s " %(eval_cats[-1],)
    
    if len(eval_cats)>0:     
        # print "Successfully finished evaluation for %d categories: %s " %(len(eval_cats),eval_cats)
        return True
    else:
        # print "No categories have been evaluated!"
        return False

# ------------------------------------------------------------------------------------------

# CONFIG VARIABLES

# Where to store the CSV report
dt = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
summary_file = "mrf_evaluate_results/mrf_report_{0}.csv".format(dt)

# Where the encoded prediction images are
original_prediction_dir = "results/initial_predictions_mrf/neural_net_2layer_25_10/prediction_images/5000sp/valid/encoded"
# The training image directory - must hold "gt_image_2"
train_dir = "results/initial_predictions_mrf/neural_net_2layer_25_10/prediction_images/5000sp/valid"
# Where to store (changing) MRF images
mrf_image_dir = "results/mrf_images/{0}".format(dt)

# Variables used purely for the CSV report
model_name = "mlp"
config_string = "YUAN'S CONFIGURATION STRING"
script_name = "mid-025"
data_set = "valid"
superpixels = 5000

# Iteration config
#iteration_list = [(0.1,1), (0.2,1), (0.5,4), (0.7,4), (1,1), (2,1), (4,1)] # not too sure how you want to do this - first number is temp and second is number of MRF updates

iteration_list = [('blur', 1), (2.0, 5), (1.5, 5), (1.0, 5), (0.8, 5), (0.6, 5), (0.4, 5), (0.2, 5)] # not too sure how you want to do this - first number is temp and second is number of MRF updates


mrf_model_dic = {}
for filename in os.listdir(original_prediction_dir):
    if fnmatch.fnmatch(filename, '*.png'):
        
        print 'Loading image:'

        image_type = np.uint16
        max_val = np.iinfo(image_type).max
        prediction_image = imread(os.path.join(original_prediction_dir, filename), as_grey=True).astype(image_type)
        image_height, image_width = prediction_image.shape[0], prediction_image.shape[1]
        image_pixel_priors = prediction_image/(float(max_val))

        #Apply pre-processing to image
        image_pixel_priors = gaussian_filter(image_pixel_priors, 5)

        # print('Bilateral Filter')
        # image_pixel_priors = denoise_bilateral(image_pixel_priors, sigma_range=0.05, sigma_spatial=15)

        image_pixel_priors_flat = image_pixel_priors.ravel()

        #
        print 'Testing - Max and Min Value'
        print(np.max(image_pixel_priors_flat))
        print(np.min(image_pixel_priors_flat))

        #===================================
        #script for initiating the MRF class
        #===================================

        print 'initialize MRF'
        predicted_labels = pixelmap()

        predicted_labels.load_superpixel_classifier_predictions(image_pixel_priors_flat, prediction_image.shape[0], prediction_image.shape[1])
        predicted_labels.set_conn_energy(0.25) #this is required to set the strength of connections ()
        predicted_labels.init_energy()

        mrf_model_dic[filename] = predicted_labels


for iter_i in iteration_list:
    print "Iteration {0}".format(iter_i)

    temperature = iter_i[0] 	# temperature
    updates = iter_i[1]	# num of times to do MRF updaates

    for update_j in xrange(updates):
		# # Read images from either the MRF image directory or original predictions directory
		# if not os.path.exists(mrf_image_dir):
		# 	print "Reading from original predictions"
		# 	working_image_dir = original_prediction_dir
		# else:
		# 	print "Reading from MRF file"
		# 	working_image_dir = mrf_image_dir

        # Make the MRF image directory if it doesn't exist already
        if not os.path.exists(mrf_image_dir):
        	os.makedirs(mrf_image_dir)

        # Read in either the orignal predictions or the current MRF images and do work on them
        print "Doing MRF work, writing out images to the MRF folder"
        # for filename in os.listdir(working_image_dir):
        # 	if fnmatch.fnmatch(filename, '*.png')
        # 		image = cv2.imread(os.path.join(working_image_dir, filename), 0)

        		# YUAN - DO YOUR WORK HERE!!!!!!!!!
        			# CHANGE `image` AS NEEDED
	

        if temperature == 'blur': #if we only want to output the blurred predictions
            for file_name, predicted_labels in mrf_model_dic.iteritems():
                print 'Blur - file_name:'
                print(file_name)
                print 'saving to file'
                image = np.reshape(predicted_labels.pixel_priors, [predicted_labels.height, predicted_labels.width])
                imsave(os.path.join(mrf_image_dir, file_name), image)

                #build configuration string
                config_string = (script_name+"Connection_strength: " + ('%.4f' % (predicted_labels.conn_energy)) + ", " + "blur")


        else: #this is actually temperature
            for file_name, predicted_labels in mrf_model_dic.iteritems():
                print 'MCMC - file_name:'
                print(file_name)
                print 'temperature:'
                print(temperature)

                predicted_labels.mcmc_rand_update(1/temperature)
                print(predicted_labels.total_energy)
                predicted_labels.mcmc_block_flip_update(1/temperature)
                print(predicted_labels.total_energy)
                predicted_labels.mcmc_rand_update(1/temperature)
                print(predicted_labels.total_energy)
                predicted_labels.mcmc_update(1/temperature)
                print(predicted_labels.total_energy)

                print 'Saving to file'
                image = np.reshape(predicted_labels.pixel_labels, [predicted_labels.height, predicted_labels.width])
                imsave(os.path.join(mrf_image_dir, file_name), image)


            #build configuration string
            config_string = script_name+"Connection_strength: " + ('%.4f' % (predicted_labels.conn_energy)) + ", temperature: " + ('%.4f' % (temperature)) + ", updates: " + ('%.1f' % (update_j))

        print "Evaluating MRF results"
        evaluate(mrf_image_dir, train_dir, summary_file, model_name, config_string, data_set, superpixels)
