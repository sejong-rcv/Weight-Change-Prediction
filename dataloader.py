import os
import json

import numpy as np
import pandas as pd

from tqdm import tqdm
from utils import *

  
def feature_extractor(DATA_DIR, split="train"):

    assert split in ["train", "test"]

    split = split.lower()

    with open(os.path.join(DATA_DIR, f'{split}_list_final.txt'), "r") as f:
        file_paths = f.readlines()

    features = []
    for file_path in tqdm(file_paths):
        path = file_path.strip("\n")
        
        try:
            with open(path, "r") as j:
                data = json.load(j)
        except:
            import pdb;pdb.set_trace()
        
        feature = calculate_distance(data[1]["landmark"], data[0]["landmark"])
        feature = np.append(feature, data[0]["label"])
        
        features.append(feature)

    col_name = list(range(468))
    col_name.append("label")

    return pd.DataFrame(features, columns=col_name)


def real_feature_extractor(real_data_paths):
    # For 3D Alignment, ICP algorithm is applied to landmark_1
    real_features = []
    for paths in tqdm(real_data_paths):
        path = paths.strip("\n")
        
        try:
            with open(path, "r") as j:
                data = json.load(j)
        except:
            import pdb;pdb.set_trace()

        landmark_1 = np.array(data[0]["landmark"]).reshape((-1, 468, 3))
        lamdmark_2 = np.array(data[1]["landmark"]).reshape((-1, 468, 3))

        # Perform the ICP algorithm
        landmark_1[0] = icp(landmark_1[0], lamdmark_2[0]) 

        real_feature = calculate_distance(landmark_1, lamdmark_2)
        real_feature = np.append(real_feature, [data[0]["distance"], data[0]["label"]])

        real_features.append(real_feature)

    col_name = list(range(468))
    col_name.append("distance")
    col_name.append("label")


    return pd.DataFrame(np.array(real_features), columns=col_name)



def real_syn_difference(file_paths):
    # For 3D Alignment, ICP algorithm is applied to landmark_1
    real_syn_gap = []

    for paths in tqdm(file_paths):
        path = paths.strip("\n")
        try:
            with open(path, "r") as j:
                data = json.load(j)
        except:
            import pdb;pdb.set_trace()

        landmark_1 = np.array(data[0]["landmark"]).reshape((-1, 468, 3))
        lamdmark_2 = np.array(data[1]["landmark"]).reshape((-1, 468, 3))

        # Perform the ICP algorithm
        landmark_1[0] = icp(landmark_1[0], lamdmark_2[0]) 

        real_feature = calculate_distance(landmark_1, lamdmark_2)
        real_syn_gap.append(real_feature)

    gap_mean = calculate_syn_real_gap(real_syn_gap)

    return gap_mean