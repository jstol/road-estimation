#!/usr/bin/env python
#
#  THE KITTI VISION BENCHMARK SUITE: ROAD BENCHMARK
#
#  Copyright (C) 2013
#  Honda Research Institute Europe GmbH
#  Carl-Legien-Str. 30
#  63073 Offenbach/Main
#  Germany
#
#  UNPUBLISHED PROPRIETARY MATERIAL.
#  ALL RIGHTS RESERVED.
#
#  Authors: Tobias Kuehnl <tkuehnl@cor-lab.uni-bielefeld.de>
#           Jannik Fritsch <jannik.fritsch@honda-ri.de>
#

import sys,os,csv
from glob import glob
import shutil
from helper import evalExp, pxEval_maximizeFMeasure, getGroundTruth
import numpy as np
import cv2 # OpenCV

class dataStructure: 
    '''
    All the defines go in here!
    '''
    
    # cats = ['um_lane', 'um_road', 'umm_road', 'uu_road']
    cats = ['um_road', 'umm_road', 'uu_road']
    calib_end = '.txt'
    im_end = '.png'
    gt_end = '.png'
    prob_end = '.png'
    eval_propertyList = ['MaxF', 'AvgPrec', 'PRE_wp', 'REC_wp', 'FPR_wp', 'FNR_wp' ] 

#########################################################################
# function that does the evaluation
#########################################################################
def main(result_dir, train_dir, summary_file, model_name, config_string, data_set, superpixels, debug = False):
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
    


#########################################################################
# evaluation script
#########################################################################
if __name__ == "__main__":

    # check for correct number of arguments.
    if len(sys.argv)!=8:
        print "Usage: python evaluateRoad.py <result_dir> <gt_dir> <summary_file> <config_string> <data_set>"
        print "<result_dir> = directory with the result propability maps, e.g., /home/elvis/kitti_road/my_results"
        print "<gt_dir> = training directory (has to contain gt_image_2)  e.g., /home/elvis/kitti_road/training"
        print "<summary_file> = sumary CSV file e.g., /home/elvis/report.csv"
        print "<model_name> = name of the algorithm used e.g., 'knn'"
        print "<config_string> = string used to identify the model's configuration e.g., '{\"k\": 1}' or 'model1'"
        print "<data_set> = data set being tested e.g., 'test' or 'valid'"
        print "<superpixels> = the number of superpixels being tested"
        sys.exit(1)
      
    # parse parameters
    result_dir = sys.argv[1]
    gt_dir = sys.argv[2]
    summary_file = sys.argv[3]
    model_name = sys.argv[4]
    config_string = sys.argv[5]
    data_set = sys.argv[6]
    superpixels = sys.argv[7]

    # Excecute main fun 
    main(result_dir, gt_dir, summary_file, model_name, config_string, data_set, superpixels)


