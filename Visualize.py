#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 20:46:34 2025

@author: emirhan
"""


import matplotlib.pyplot as plt
import random

def visualizer( ds_pixel_data , number):
    
    for i in range(number):
      plt.imshow(ds_pixel_data[random.randint(0, len(ds_pixel_data) - 1)])
      plt.axis('off')  
      plt.show()