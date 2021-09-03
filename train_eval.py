import os
import cv2
import glob
import json
import random

import copy
import numpy as np
import pandas as pd

from tqdm import tqdm

import open3d as o3d
from translation import icp
from dataloader import *

from pycaret.classification import *

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from pycaret.utils import check_metric

random.seed(0)

load_model = None

DATA_DIR = "./"
REFINE_DIR = os.path.join(DATA_DIR, "Real_Dataset")
REAL_DIR = os.path.join(DATA_DIR, "Real_Dataset")

refine_list = glob.glob(REFINE_DIR + "/refine_*.json")
real_data_list = glob.glob(REAL_DIR + "/real_*.json")

load_pretrained = None

def train_eval():

  ### Data Loader & Feature Extractor ###
  train = feature_extractor(DATA_DIR, split="train")
  test = feature_extractor(DATA_DIR, split="test")

  real_data = real_feature_extractor(real_data_list)

  ### calculate synthetic - real difference ### 
  gap_mean = real_syn_difference(refine_list)
  train_refine = refine_data(train, gap_mean)
  test_refine = refine_data(test, gap_mean)


  #### train ####
  if load_pretrained == None:  
    clf = setup(data = train_refine, target = 'label', session_id = 123)
    set_config('seed', 999)

    refine_data_best = compare_models(n_select = 15)
    refine_blended = blend_models(estimator_list = refine_data_best, fold = 5)
    model = finalize_model(refine_blended)  

  else :
    model = load_pretrained(load_pretrained)
  

  ### Evaluation ###
  predictions = predict_model(model, data = test_refine)
  real_prediction = predict_model(model, data = real_data).astype("float")


  print("Real Data Accuracy")
  print(f"Accuracy : {check_metric(real_prediction['label'], real_prediction['Label'], metric='Accuracy')}")
  



if __name__ == '__main__':
  train_eval()