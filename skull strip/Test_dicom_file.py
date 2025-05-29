# -*- coding: utf-8 -*-
"""
Created on Thu May  2 18:14:11 2019

@author: cand07
"""

import torch
import dicom2nifti


from multiprocessing import freeze_support
from HD_BET.checkpoint_download import maybe_download_parameters
from HD_BET.hd_bet_prediction import get_hdbet_predictor, hdbet_predict


if __name__ == '__main__':
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    freeze_support()
#     dicom_directory = "./UPENN-GBM-00001"
#     output_folder = "./nifti/"
    
#     dicom2nifti.convert_directory(dicom_directory, output_folder)
    
#     import pydicom
#     print(pydicom.__version__)
    

    disable_tta = False
    device = "cpu"
    verbose = True
    save_bet_mask = True
    no_bet_image = False
    inp = "./nifti/"
    out = "./stripped/"
    
    maybe_download_parameters()
        
    predictor = get_hdbet_predictor(
        use_tta=not disable_tta,
        device=torch.device(device),
        verbose=verbose
    )
    
    hdbet_predict(inp, out, predictor, keep_brain_mask=save_bet_mask,
                  compute_brain_extracted_image=not no_bet_image)

