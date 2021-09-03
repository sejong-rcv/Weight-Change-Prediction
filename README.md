## Weight-Change-Prediction-for-Automated-Depression-Diagnosis

This code is for ***Weight-Change-Prediction-for-Automated-Depression-Diagnosis*** submitted to ICCAS 2021.

## Install

```bash
git clone https://github.com/sejong-rcv/Weight-Change-Prediction.git
cd Weight-Change-Prediction
```

## Datasets

We collected ID photos and selfies of celebrities, and generated synthetic data using the *PhotoWorks* tool. 
The dataset we released is the 3D landmarks extracted from crawled images.

Synthetic data are processed into five stages for weight change.

- level_1 : Excessive Weight Loss
- level_2 : Little Weight loss
- level_3 : Original Image
- level_4 : Little Weight Gain (eg. Swelling)
- level_5 : Excessive Weight Gain

```python
./ICCAS_2021/
├── Synthetic_Dataset/
│   ├── 001_1_landmark.json # person ID 001's level(1 & 3) landmark
│   ├── 001_2_landmark.json # person ID 001's level(2 & 3) landmark
│   ... 
│   └── 500_5_landmark.json # person ID 500's level(5 & 3) landmark
│
├── Real_Dataset/
│   ├── real_002_landmark.json
│   ├── real_001_landmark.json
│    ... 
│   ├── refine_012_landmark.json
│   └── refine_013_landmark.json
│
├── train_list_final.txt          # train file
└── test_list_final.txt           # test file 
```
The weight change level was randomly sampled and used for training, so it provides a path to the train and test files as a `txt` file.

`json` files are configured as follows:
```python
[
  { "level"     # Synthetic Image's Level for comparison with the original
    "landmark"  # Landmark of Synthetic Image (468, 3)
    "label"     # 1 or 0 : When there's a weight change, label is 1
  }, 
  {
    "level"     # Original Image Level
    "landmark"  # Landmark of Original Image (468, 3)
    "label"     # the same level as the Synthetic image in same file
  }
]
```



## Implementation

We provide [colab tutorial](https://colab.research.google.com/drive/15Tuyp9qlGEwMaVemglWTclP_NSdjY3Fy?usp=sharing), and full guidance. There are also tutorials for training the model, extracting features from landmarks, and visualization for 3D Face Landmark. (you can check the colab tutorial code in [ipynb file](Weight_Change_Prediction_for_Automated_Depression_Diagnosis))


Set the data directory and pretrained model path –
```python
DATA_DIR = "/ICCAS_2021" # Data Directory
load_model = None # if you want to use pretrained model, insert the .pkl file path
```

To train and evaluate the model, run this file –

```bash
python train_eval.py
```

## Performance

| Model | Accuracy |
| :--- | :---: |
| Light Gradient Boosting (LightGBM)   | 93.00 |
| K Nearest Neighbor (KNN)             | 92.75 |
|Random Forest (RF)                    | 92.25 | 
|Linear Discriminant Analysis (LDA)    | 89.25 | 
|Decision Tree (DT)                    | 88.00 | 
|LogisticRegression (LR)               | 87.00 | 
|Quadratic Discriminant Analysis (QDA) | 69.00 | 


| Depth | Accuracy |
| :---: |  :---:   |
| 30cm  |   0.7    |
| 30cm  |   0.7    |
| 45cm  |   0.9    |

## References

**Weight-Change-Prediction-for-Automated-Depression-Diagnosis (ICCAS)**

*Juyoung Hong, Jeongmin Shin, Yujin Hwang, Jeongmin Lee, and Yukyung Choi*

```bash
@inproceedings{weight_change_prediction,
  author = {Juyoung Hong, Jeongmin Shin, Yujin Hwang, Jeongmin Lee, and Yukyung Choi},
  title = {Weight-Change-Prediction-for-Automated-Depression-Diagnosis},
  booktitle = {International Conference of Contral, Automation and Systems(ICCAS)},
  year = {2021},
}
```
