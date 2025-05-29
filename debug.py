#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 11:08:58 2025

@author: ai1
"""

import os
import glob2
import pandas as pd
import numpy as np
import cv2
import SimpleITK as sitk
from skimage.exposure import match_histograms


from tqdm import tqdm  # Make sure this is imported correctly



from utils.utils import stack_3,  standardization, show_features, normalize
from model.model import CNN_Model, FineTuned_CNN_Model, CNN_3D
from keras.utils import to_categorical
from keras.applications.inception_v3 import preprocess_input
import keras
import tensorflow as tf



# ---- CONFIGURATION ----
SRC_FOLDER = "dataset/ready"   # Folder with 3D CT .npy files
DST_FOLDER = "dataset/mri_features"    # Folder to save extracted features
IMG_SIZE = 256                              # CNN input size


# ---- INITIALIZE CNN MODEL ----
cnn_model = FineTuned_CNN_Model(IMG_SIZE, preprocess_input)
cnn_model.summary()
# ---- ENSURE DESTINATION EXISTS ----
os.makedirs(DST_FOLDER, exist_ok=True)

# ---- FEATURE EXTRACTION LOOP ----
for filename in tqdm(os.listdir(SRC_FOLDER)):
    if not filename.endswith('.npy'):
        continue

    patient_id = filename.split('.')[0].split("(")[0]
    input_path = os.path.join(SRC_FOLDER, filename)
    output_path = os.path.join(DST_FOLDER, f"{patient_id}.npy")

    # Skip if already processed
    if os.path.exists(output_path):
        continue

    try:
        # Load & preprocess image
        ct = np.load(input_path)                   # (D, H, W)
        ct = standardization(ct)                   # normalize
        ct = stack_3(ct)                            # (D, H, W, 3)
        ct = tf.image.resize(ct, (IMG_SIZE, IMG_SIZE))  # (D, IMG_SIZE, IMG_SIZE, 3)
        ct = np.array(ct, dtype=np.float32)

        # Extract features from each slice
        features = cnn_model.predict(ct, verbose=0)  # (D, 2048)

        # Save extracted features
        np.save(output_path, features)
    except Exception as e:
        print(f"Failed to process {filename}: {e}")