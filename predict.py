#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 12:06:47 2025

@author: ai1
"""

import numpy as np
from multiprocessing import freeze_support
from utils.preprocessing import preprocess_mri_file
from utils.preprocess_his import preprocess_his_file
from config import Config
from utils.utils import stack_3,  standardization, show_features, normalize
from model.model import CNN_Model, FineTuned_CNN_Model, CNN_3D
from keras.utils import to_categorical
from keras.applications.inception_v3 import preprocess_input
import keras
import tensorflow as tf
from tensorflow.keras.models import load_model
import inference
from tensorflow.keras import backend as K

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from keras.applications.inception_v3 import preprocess_input
from utils.preprocessing import preprocess_mri_file
from utils.preprocess_his import preprocess_his_file
from utils.utils import standardization, stack_3
from model.model import FineTuned_CNN_Model
import inference
from config import Config
import nibabel as nib
import numpy as np
from PIL import Image
import os
import json

def predict(mri_path: str, histo_path: str, clinical_data: list):
    
    
    
    print(histo_path)
    # Histopathology feature extraction
    his_np = preprocess_his_file(histo_path)
    his_input = np.expand_dims(his_np, axis=0) 
    conf = Config()
    proc = preprocess_mri_file(conf)
    
    output_path = "uploads/send/mri.png"
    img = nib.load(mri_path)
    data = img.get_fdata()  # (H, W, D)

    # Z ekseninden (3. boyut) tam ortadaki slice'ı seç
    middle_index = data.shape[2] // 2
    middle_slice = data[:, :, middle_index]  # (H, W)

    # Normalize et (0-255)
    norm_slice = ((middle_slice - np.min(middle_slice)) /
                  (np.max(middle_slice) - np.min(middle_slice)) * 255).astype(np.uint8)

    img_pil = Image.fromarray(norm_slice, mode='L')  # 'L' = 8-bit grayscale
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img_pil.save(output_path)
    
    
    

    IMG_SIZE = 128
    try:
        # MRI preprocessing
        mri_np = proc(mri_path)
        ct = standardization(mri_np)
        ct = stack_3(ct)
        ct = tf.image.resize(ct, (IMG_SIZE, IMG_SIZE))
        ct = np.array(ct, dtype=np.float32)

        # MRI feature extraction
        mri_feature_model = 'model/best_model2.keras'
        cnn_model = FineTuned_CNN_Model(IMG_SIZE, preprocess_input, mri_feature_model)
        mri_features = cnn_model.predict(ct, verbose=0)
        mri_input = np.expand_dims(mri_features, axis=0)

       

        print("Clinical data" , clinical_data)
        # Clinical input
        clinical_input = np.array([clinical_data], dtype=np.float32)

        X_sample = [mri_input, his_input, clinical_input]

        # Load main model & predict
        model = load_model('plot/best_model.keras')
        predictions = model.predict(X_sample)

        death_pred, grade_pred, mortality_pred, gt_pred = predictions
        
        print("death")
        print(death_pred)
        
        print("grade")
        print(grade_pred)
        
        print("morta")
        print(mortality_pred)
        
        print("gt")
        print(gt_pred)

        # SHAP açıklamaları
        background = [
            np.load('test/mri_b.npy'),
            np.load('test/his_b.npy'),
            np.load('test/clinic_b.npy')
        ]

        shap_values, [mri_pct, clinical_pct, his_pct] = inference.shapley(
            model, X_sample, background=background, output_name='death'
        )
        
        values = [float(gt_pred[0][i]) for i in range(3)]  # 3 değer alınıyor
        max_index = values.index(max(values)) + 2
        
        values = [float(grade_pred[0][i]) for i in range(2)]  # 3 değer alınıyor
        max_index_grade = values.index(max(values))
        
        
        
        if(max_index_grade == 1):
            max_index = 4
            grade = "Glioblastoma"
        else:
            grade = "Low Grade Glioma"   

        K.clear_session()
        
        filename = os.path.basename(histo_path) 
    
        save_dir = 'uploads/send'
        
        result = {
            "death": [
                float(death_pred[0][0]),
                float(death_pred[0][1]),
                float(death_pred[0][2])
            ],
            
            "grade": grade,
            "mortality": float(mortality_pred[0][0]),
            "gt": max_index,
            "shap": {
                "mri": f"{mri_pct:.2f}",
                "clinical": f"{clinical_pct:.2f}",
                "histology": f"{his_pct:.2f}"
             },
            "project_id": filename[:12]
            
            }
        
        os.makedirs(save_dir, exist_ok=True)
    
        filename = "result.json"
        save_path = os.path.join(save_dir, filename)
    
        with open(save_path, "w") as f:
            json.dump(result, f, indent=4)

        return {
            "death": [
                float(death_pred[0][0]),
                float(death_pred[0][1]),
                float(death_pred[0][2])
            ],
            
            "grade": grade,
            "mortality": float(mortality_pred[0][0]),
            "gt": max_index,
            "shap": {
                "mri": f"{mri_pct:.2f}",
                "clinical": f"{clinical_pct:.2f}",
                "histology": f"{his_pct:.2f}"
             },
            "project_id": filename[:12]
            
            }

    except Exception as e:
        return {"error": str(e)}


# if __name__ == "__main__":
#     freeze_support()
#     conf = Config()
#     proc = preprocess_mri_file(conf)
    
    
#     IMG_SIZE = 128
#     mri_file = "test/10.000000-AX%20T1%20POST%20GD%20FLAIR-15937"
#     his_path = ''
#     b_path = 'test/'
#     mri_np = proc(mri_file)
    
#     #his_np = process_his_file(his_path)
    
#     mri_feature_model = 'model/best_model2.keras'
#     cnn_model = FineTuned_CNN_Model(IMG_SIZE, preprocess_input, mri_feature_model)

#     try:
#         # Load & preprocess image
#         ct = mri_np
#         ct = standardization(ct)                   # normalize
#         ct = stack_3(ct)                            # (D, H, W, 3)
#         ct = tf.image.resize(ct, (IMG_SIZE, IMG_SIZE))  # (D, IMG_SIZE, IMG_SIZE, 3)
#         ct = np.array(ct, dtype=np.float32)

#         # Extract features from each slice
#         mri_features = cnn_model.predict(ct, verbose=0)  # (D, 2048)

#         # Save extracted features
#         np.save('test/predict/mri_feature.npy', features)
        
#     except Exception as e:
#         print(f"Failed to process {mri_file}: {e}")
#     mri_input = np.expand_dims(mri_features, axis= 0)
#     his_input = np.random.rand(1, 2500, 512)
#     clinical_input = np.random.rand(1, 2)
    
    

#     X_sample = [mri_input, his_input, clinical_input]
    
  
#     main_model = 'plot/best_model.keras'
    
#     model = load_model(main_model)
#     predictions = model.predict(X_sample)

#     death_pred, grade_pred, mortality_pred, gt_pred = predictions

#     print("Death Prediction:", death_pred)
#     print("Grade Prediction:", grade_pred)
#     print("Mortality Prediction:", mortality_pred)
#     print("GT Prediction:", gt_pred)
    
#     K.clear_session()

#     background = [np.load(b_path+ 'mri_b.npy'), np.load(b_path + 'his_b.npy'), np.load(b_path + 'clinic_b.npy')]
    
#     shap_values, [mri_pct, clinical_pct, his_pct] = inference.shapley(model, X_sample, background=background, output_name = 'death')

#     input_names = ["mri", "clinical", "histology"]
#     print(f"  {input_names[0]}: {mri_pct:.2f}%")
#     print(f"  {input_names[1]}: {clinical_pct:.2f}%")
#     print(f"  {input_names[2]}: {his_pct:.2f}%")