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


def plot_loss(history, name, title = None):
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
    
    fname = name+'_loss.png'
    plt.savefig(fname)
    
  
    
def confusion_matrix_binary(y_true, y_pred, threshold):
    """
    Given the threshold, it calculates the Confusion Matrix. 
    :params y_true
    :params y_pred 
    :params threshold
    """

    n_shape = y_pred.shape[0]
    _y_pred = np.argmax(y_pred, axis =1)
    _y_true = np.zeros([n_shape, 1])
    _y_pred = np.zeros([n_shape, 1])
    for i in range(n_shape):
        if y_true[i,0]:
            _y_true[i] = 0
        else:
            _y_true[i] = 1

        if y_pred[i,0] > threshold:
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
    


def plot_roc_curve_binary(nb_classes, Y_test, y_pred):

    """
    Takes the true and the predicted probabilities and generates the ROC curve
    :param y_true true values 
    :param y_pred predicted values 
    """

    fpr = dict()
    tpr = dict()
    thr = dict()
    roc_auc = dict()
    # nb_classes = Y_test.shape[1]  ## TODO: check this
    
    for i in range(nb_classes):
        fpr[i], tpr[i], thr[i] = roc_curve(Y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    fpr = fpr[0]
    tpr = tpr[0]
    thr = thr[0]
    print(thr)
    
    "optimal cut off"
    opt_idx = np.argmax(tpr - fpr)
    
    # opt_idx = 5
    
    opt_fpr = fpr[opt_idx]
    opt_tpr = tpr[opt_idx]
    opt_thr = thr[opt_idx]
    print(opt_thr)
    # "optimal cut off - second way"
    # gmeans = np.sqrt((tpr*(1-fpr)))
    # "locate the index of the largest gmean"
    # idx = np.argmax(gmeans)
    # opt_thr = thr[idx]
    
       
      
    plt.figure()  
    plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve (area = %0.2f)' % roc_auc[1])
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.plot(opt_fpr,opt_tpr,'ko', label='cut-off value' )
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.grid(True) 
    plt.legend(loc="lower right")
    plt.savefig('roc.png')
    
    return roc_auc, opt_thr


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def saveResults(y_pred, y_test, nb_classes, filename):

    AUC, th = plot_roc_curve_binary(nb_classes, y_test, y_pred)

    cm = confusion_matrix_binary(y_test, y_pred, th)
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
    NPV = TN/(TN + FN)
    # F1_score = (2*TP)/((2*TP) + FP + FN)
    F1_score = (2*precision*recall)/(precision+recall)
    F2_score = (5*precision*recall)/(4*precision+recall)
    AUC = AUC[0]

    "maximizing the Youden`s index"   
    for_cutoff = sensitivity + specificity - 1
    
    " save output to a text file... "
    filename = filename + str(th) +'.txt'
    text_file = open(filename, "w")
    text_file.write("accuracy: %s \n" % acc)
    text_file.write("precision (PPV): %s \n" % precision)  
    text_file.write("sensitivity/recall: %s \n" % sensitivity)  
    text_file.write("specificity: %s \n" % specificity)
    text_file.write("NPV: %s \n" % NPV)
    text_file.write("F1_score: %s \n" % F1_score)
    text_file.write("F2_score: %s \n" % F2_score)
    text_file.write("AUC: %s \n" % AUC)
    text_file.write("TPR: %s \n" % TPR)
    text_file.write("FPR: %s \n" % FPR)
    text_file.write("for_cutoff: %s \n" % for_cutoff)
    text_file.close()
