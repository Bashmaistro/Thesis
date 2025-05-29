# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 19:21:09 2021

@author: S.Candemir (candemirsema@gmail.com)


This script is experimental codes of paper " Detecting and Characterizing Inferior 
Vena Cava Filters on Abdominal Computed Tomography with Data-driven Computational 
Frameworks" by S.Candemir et al. You may use the codes only for research. 
Plese cite the paper if you use any part of the codes.

"""

from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
from keras.models import Model
from tensorflow.keras.layers import Input
from model.model import RNN_Model, CNN_RNN_Model, CNN_3D, Multimodel, build_his_block_2, build_rnn_block, build_his_block
from keras.losses import CategoricalFocalCrossentropy
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import shap

def plot_loss(history, path, title = None):
    '''
    plot history for loss   
    '''
    
    fig = plt.figure()
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    if title is not None:
        plt.title(title)
    plt.grid(True) 
    
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.legend(['Train', 'Val'], loc=0)
    
    fname = path +'_loss.png'
    plt.savefig(fname)
    
def shapley(model, X_sample, background, output_name, input_names=None, sample_size=50, use_gradient=True ):
        """
        Performs SHAP analysis for a specific output layer in a multi-input, multi-output Keras model.

        Parameters:
        - model: Trained Keras model.
        - test_gen: Data generator yielding ([input1, input2, ...], {output_name: label, ...}).
        - output_name: Name of the model output layer to analyze (e.g., 'death').
        - input_names: Optional list of input block names for display.
        - sample_size: Number of samples to use for SHAP (keep small to reduce compute).
        - use_gradient: If True, uses GradientExplainer instead of DeepExplainer (for better compatibility).

        Returns:
        - shap_values: SHAP values for each input.
        - sample_inputs: Input sample used for SHAP analysis.
        """
        # for i, layer in enumerate(self.model.layers):
        #     try:
        #         print(i, layer.name, layer.output.shape)
        #     except AttributeError:
        #         print(i, layer.name, "InputLayer (no output shape)")
        
        # try:
        # Step 1: Get the output of the fusion layer (after all concatenations)
        fusion_output = model.layers[-8].output


        # Step 2: Define the predictor head that maps from fused vector to a specific output
        fused_input = Input(shape=(fusion_output.shape[-1],), name="fused_input")
         
        x = model.get_layer("dense1")(fused_input)
        x = model.get_layer("dense2")(x)
        x = model.get_layer(output_name)(x)
        # Create a new model up to the desired layer
        model = Model(inputs=model.inputs, outputs=fusion_output, name="truncated_model")
        predictor_model = Model(inputs=fused_input, outputs=x)
        feature_model = model
        print(f"\n========== SHAP Analysis for '{output_name}' ==========")

        # Step 3: Generate fused inputs from your sample data
        X_fused = feature_model.predict(X_sample)

        background2 = feature_model.predict(background)
        # Step 4: Run SHAP KernelExplainer
        explainer = shap.DeepExplainer(predictor_model, background2)
        shap_values = explainer.shap_values(X_fused, check_additivity=False)
        shap_block = np.mean([np.abs(class_shap) for class_shap in shap_values], axis=0)
        if input_names is None:
            input_names = ["mri", "clinical", "histology"]
        mri_dim = 128
        clinical_dim = 2
        his_dim = 258 - mri_dim - clinical_dim
        
        mri_shap = shap_block[:mri_dim, :]
        clinical_shap = shap_block[mri_dim:mri_dim + clinical_dim, :]
        his_shap = shap_block[mri_dim + clinical_dim:, :]

        print("MRI SHAP shape:", mri_shap.shape)
        print("Clinical SHAP shape:", clinical_shap.shape)
        print("Histology SHAP shape:", his_shap.shape)

        num_classes = shap_values.shape[1]

        print("Average SHAP contribution percentages per modality for each output class:")
        print(clinical_shap)
        
        # Sum SHAP per modality, summed over all samples
        mri_sum = np.sum(mri_shap)
        clinical_sum = np.sum(clinical_shap)
        his_sum = np.sum(his_shap)
        
        total_sum = mri_sum + clinical_sum + his_sum
        
        if total_sum == 0:
            mri_pct = clinical_pct = his_pct = 0.0
        else:
            mri_pct = 100 * mri_sum / total_sum
            clinical_pct = 100 * clinical_sum / total_sum
            his_pct = 100 * his_sum / total_sum
        

        # Optional: SHAP summary plot (use smaller size if needed)
        #shap.summary_plot(shap_values, X_fused, feature_names=None)
        
        # except Exception as e:
        #     print(f"SHAP analysis failed for output '{output_name}': {e}")
        #     return None, None
        
        return shap_values, [mri_pct, clinical_pct, his_pct]

    
def confusion_matrix_binary(y_true, y_pred, threshold):
    """
    Given the threshold, it calculates the Confusion Matrix. 
    :params y_true
    :params y_pred 
    :params threshold
    """

    n_shape = y_pred.shape[0]
    _y_pred = np.argmax(y_pred, axis = 0)
    _y_true = np.zeros([n_shape, 1])
    _y_pred = np.zeros([n_shape, 1])
    for i in range(n_shape):
        if y_true[i]:
            _y_true[i] = 0
        else:
            _y_true[i] = 1

        if y_pred[i] > threshold:
            _y_pred[i] = 0
        else:
            _y_pred[i] = 1

    cm = confusion_matrix(_y_true, _y_pred)
    np.savetxt('Confusion_matrix.txt', cm, fmt='%.1f')
    
    TN = cm[0][0];
    FP = cm[0][1]; 
    FN = cm[1][0];  
    TP = cm[1][1];
    
    return cm
    


def plot_roc_curve_binary(Y_test, y_pred):
    """
    Takes the true binary labels and the predicted probabilities and generates the ROC curve.
    :param Y_test: true binary labels (0 or 1)
    :param y_pred: predicted probabilities (between 0 and 1)
    """
    # Calculate False Positive Rate, True Positive Rate, and Thresholds
    fpr, tpr, thresholds = roc_curve(Y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    # Find the optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    print(f"Optimal Threshold: {optimal_threshold}")
    print(f"Optimal FPR: {fpr[optimal_idx]}, Optimal TPR: {tpr[optimal_idx]}")

    # "optimal cut off - second way"
    # gmeans = np.sqrt((tpr*(1-fpr)))
    # "locate the index of the largest gmean"
    # idx = np.argmax(gmeans)
    # opt_thr = thr[idx]
    
       
      
    # plt.figure()  
    # plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve (area = %0.2f)' % roc_auc[1])
    # plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.plot(fpr[optimal_idx],tpr[optimal_idx],'ko', label='cut-off value' )
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic')
    # plt.grid(True) 
    # plt.legend(loc="lower right")
    # plt.savefig('roc.png')
    
    # return roc_auc, optimal_threshold
   # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', label=f'Optimal Threshold = {optimal_threshold:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    return roc_auc, optimal_threshold
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def saveResults(y_pred, y_test, nb_classes, filename):

    AUC, th = plot_roc_curve_binary(y_test, y_pred)

    cm = confusion_matrix_binary(y_test, y_pred, th)
    print(cm)

    TN = cm[0][0];
    FP = cm[0][1]; 
    FN = cm[1][0];  
    TP = cm[1][1];

    TPR = TP/(TP + FN)
    FPR = FP/(FP + TN)
    
    acc = (TP + TN)/(TP + TN + FN + FP)
    precision = TP/(TP + FP) 
    sensitivity = TP/(TP + FN)  # number of sick people in the population
    recall = sensitivity  # number of sick people in the population
    specificity = TN/(TN + FP)

    # F1_score = (2*TP)/((2*TP) + FP + FN)
    F1_score = (2*precision*recall)/(precision+recall)
    F2_score = (5*precision*recall)/(4*precision+recall)
    

    "maximizing the Youden`s index"   
    for_cutoff = sensitivity + specificity - 1
    
    " save output to a text file... "
    filename = filename + str(th) +'.txt'
    text_file = open(filename, "w")
    text_file.write("accuracy: %s \n" % acc)
    text_file.write("precision (PPV): %s \n" % precision)  
    text_file.write("sensitivity/recall: %s \n" % sensitivity)  
    text_file.write("specificity: %s \n" % specificity)
    text_file.write("F1_score: %s \n" % F1_score)
    text_file.write("F2_score: %s \n" % F2_score)
    text_file.write("AUC: %s \n" % AUC)
    text_file.write("TPR: %s \n" % TPR)
    text_file.write("FPR: %s \n" % FPR)
    text_file.write("for_cutoff: %s \n" % for_cutoff)
    text_file.close()