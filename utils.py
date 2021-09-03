import os
import numpy as np
import pandas as pd

import cv2
import copy
from tqdm import tqdm

import open3d as o3d


def icp(source_face, target_face):
  
  source = o3d.geometry.PointCloud()
  target = o3d.geometry.PointCloud()

  source.points = o3d.utility.Vector3dVector(source_face[:,:3])
  target.points = o3d.utility.Vector3dVector(target_face[:,:3])

  threshold = 0.3 #icp threshold
  trans_init = np.asarray([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0], [0.0, 0.0, 0.0, 1.0]])
  reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100000))
  
  source_tmp = copy.deepcopy(source)
  source_tmp.transform(reg_p2p.transformation)
  
  return np.asarray(source_tmp.points)


def calculate_syn_real_gap(real_syn_gap):
    # Calculte the real data view-point error mean
    real_error_mean = 0
    for i in real_syn_gap:
        real_error_mean += i
    real_error_mean = real_error_mean / len(real_syn_gap)
    return real_error_mean


def refine_data(data, real_error_mean):

    mod_data = data.iloc[:,:-1] + real_error_mean
    mod_data["label"] = data["label"]

    return mod_data


def calculate_distance(landmark_1, landmark_2):  

    difference = np.array(landmark_1) - np.array(landmark_2)
    difference = difference.reshape(468,3)
    distance = np.sqrt(np.sum(difference**2, axis=1))
    
    return distance