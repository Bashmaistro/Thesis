#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 20:40:14 2025

@author: emirhan
"""

import pydicom
import numpy as np

def white_filter(dicom_file):
    
    ds = pydicom.dcmread(dicom_file)
    ds_pixel_array = ds.pixel_array
    
    

    data = []
    for i in range(ds_pixel_array.shape[0]):
    
      if ds_pixel_array[i].mean() < 230:
        data.append(ds_pixel_array[i])

    array = np.array(data) 

    ds.set_pixel_data(array, 
                      photometric_interpretation="RGB", 
                      bits_stored=ds.BitsStored)
    
    return ds