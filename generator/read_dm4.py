# -*- coding: utf-8 -*-

from __future__ import division

import os
import numpy as np
import matplotlib.pyplot as plt
import ncempy.io.dm as dm
import cv2
from tqdm import tqdm

class ReadDm4:

    def normalise(self, array:np.ndarray) -> np.ndarray:
        
        """ Normalise data between 0 and 1 """
        
        normalised_array = ((array-np.min(array))/(np.max(array)-np.min(array)))
        
        return normalised_array

    def lower_upper_bound_cut(self, array:np.ndarray) -> np.ndarray:
        
        """ find values that cut the ends off the histogram 
        of the image,in order to make it more viewable and 
        more spread in the 0 to 1 range. """
        
        flatten = np.ndarray.flatten(array) # flatten the array to 1-d
        
        cut_index_lower = int(len(flatten)*0.00001) #find index of value where 0.001% of data is under
        
        cut_index_upper = int(len(flatten)*0.99999) #find index of value where 0.001% of data is above
        
        flatten_sort = np.sort(flatten) # sort the flattened array from lowest to highest
        
        lower_bound_value = flatten_sort[cut_index_lower] #find lower bound value for cut off
        
        upper_bound_value = flatten_sort[cut_index_upper] # find upper bound value for cut off
        
        return lower_bound_value,upper_bound_value

    # def get_resolution(resolution,unit):
        
    #     if unit == 'nm':
            
    #         resolution = resolution/1000
            
    #     return resolution

    def get_image_from_dm4(self, path:str) -> np.ndarray:

        """
        Reads a .dm file, gets metadata and image.
        
        Arguments:
            
            "path": path to .dm file
            
        Returns:
            
            "img_blur": image from .dm file on [0, 255] interval
            "resolution": pixel size
            "unit": unit of resolution, normally μm or nm.
            
        """
                        
        dm3_file = dm.dmReader(path, verbose=False ) #read the dm3 file


        img = np.array(dm3_file['data']) # get the image array from dm3 file
        
        resolution, unit = dm3_file['pixelSize'][0],dm3_file['pixelUnit'][0]

        lower_bound_value,upper_bound_value = self.lower_upper_bound_cut(img) # get the lower and upper bound values for histogram cutting
        
        img[img<lower_bound_value] = np.random.uniform(lower_bound_value,upper_bound_value,len(img[img<lower_bound_value])) # cut histogram lower
        
        img[img>upper_bound_value] = np.random.uniform(lower_bound_value,upper_bound_value,len(img[img>upper_bound_value])) # cut histogram upper

        img = self.normalise(img) # normalise the new array
        
        img_blur = cv2.blur(np.squeeze(img),(5,5))
        
        # img_blur = cv2.resize(img_blur,
        #                         dsize=(2048,2048),
        #                         interpolation=cv2.INTER_NEAREST)

        return img_blur*255, resolution, unit