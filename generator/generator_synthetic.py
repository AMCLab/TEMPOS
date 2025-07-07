from .grid import Grid
from .generator_utils import GeneratorUtils
from .particles import Particles
from .annotations import GetAnnotations
from generator.read_dm4 import ReadDm4
from PyQt6 import QtCore
from tqdm import tqdm
from typing import Tuple
import os
from skimage import exposure
import skimage
import colorsys
import numpy as np
import cv2
import datetime
from numba import jit
import albumentations as A
import random
from collections import Counter

@jit
def _add_arrays(array1:np.ndarray, array2:np.ndarray) -> np.ndarray:

    added_array = np.add(array1, array2)

    return added_array

@jit
def _create_zeros_array(size:tuple, dtype:np.dtype) -> np.ndarray:

    zeros_array = np.zeros(size, dtype)

    return zeros_array

@jit
def _create_zeros_like_array(array:tuple) -> np.ndarray:

    zeros_array = np.zeros_like(array)

    return zeros_array

@jit
def _create_empty_like_array(array:tuple) -> np.ndarray:

    empty_array = np.zeros_like(array)

    return empty_array

class GenerateSyntheticThread(QtCore.QThread):
    
    finished_signal = QtCore.pyqtSignal(dict, dict, dict, dict, int)

    progress = QtCore.pyqtSignal(int)

    def __init__(self, number_of_images,  settings, count, folder, data_train, data_val, data_train_coco, data_val_coco):
       
        QtCore.QThread.__init__(self)

        self.syntheticGenerator = SyntheticGenerator()

        self.syntheticGenerator.progress.connect(self.send_progress)

        self.number_of_images = number_of_images
        self.settings = settings
        self.count = count
        self.folder = folder
        self.data_train = data_train
        self.data_val = data_val 
        self.data_train_coco = data_train_coco 
        self.data_val_coco = data_val_coco

    def run(self):
        tic = datetime.datetime.now()
        print('tic ', tic)
        data_train, data_val, data_train_coco, data_val_coco, count, _, _ = self.syntheticGenerator.generate_synthetic_data(self.number_of_images, self.settings, 
                                                                                                                                        self.count, self.folder, self.data_train, self.data_val, 
                                                                                                                                        self.data_train_coco, self.data_val_coco)

        self.finished_signal.emit(data_train, data_val, data_train_coco, data_val_coco, count) 
        
        tac = datetime.datetime.now()
        print('tac ', tac)

        print('process with numba took ' + str(tac - tic))
    def send_progress(self, value):
        
        self.progress.emit(value)

class PlotSyntheticThread(QtCore.QThread):
    
    finished_signal = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, number_of_images, settings, count, folder, data_train, data_val, data_train_coco, data_val_coco):
       
        QtCore.QThread.__init__(self)

        self.syntheticGenerator = SyntheticGenerator()

        self.number_of_images = number_of_images
        self.settings = settings
        self.count = count
        self.folder = folder
        self.data_train = data_train
        self.data_val = data_val 
        self.data_train_coco = data_train_coco 
        self.data_val_coco = data_val_coco

    def run(self):

        _, _, _, _, _, final_image, _ = self.syntheticGenerator.generate_synthetic_data(self.number_of_images, self.settings, 
                                                                                        self.count, self.folder, self.data_train, self.data_val, 
                                                                                        self.data_train_coco, self.data_val_coco)
           
        self.finished_signal.emit(final_image) 


class SyntheticGenerator(QtCore.QObject):

    progress = QtCore.pyqtSignal(int)

    def apply_mask(self, image: np.ndarray, mask: np.ndarray, 
                   color: list, alpha=0.2) -> np.ndarray:
        """
        Apply the given mask to the image.
        """
        
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                    image[:, :, c] *
                                    (1 - alpha) + alpha * color[c] * 255,
                                    image[:, :, c])
        return image

    def plot_mask(self, final_lacey_carb:np.ndarray, final_regions: dict) -> np.ndarray:

        final_lacey_carb_3d = np.dstack((final_lacey_carb, final_lacey_carb, final_lacey_carb))
        
        mask = np.zeros([final_lacey_carb.shape[0], final_lacey_carb.shape[1], len(final_regions)], dtype=np.uint8)

        for i, p in enumerate(final_regions):

            rr, cc = skimage.draw.polygon(final_regions[p]['shape_attributes']['all_points_y'], final_regions[p]['shape_attributes']['all_points_x'])
            unwnated_index_rr = np.where(np.logical_or(rr<0, rr>=mask.shape[0]))
            rr = np.delete(rr, unwnated_index_rr)
            cc = np.delete(cc, unwnated_index_rr)
            unwnated_index_cc = np.where(np.logical_or(cc<0, cc>=mask.shape[0]))
            rr = np.delete(rr, unwnated_index_cc)
            cc = np.delete(cc, unwnated_index_cc)

            mask[rr, cc, i] = 1    

        hsv = [(i / mask.shape[2], 1, 1.0) for i in range(mask.shape[2])]
        color = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        
        for i in range(mask.shape[2]):

            image = self.apply_mask(final_lacey_carb_3d, mask[:,:,i], color[i])
        
        return image

    def create_lacey_carbon(self, mag: np.ndarray, img_size: int, 
                            inital_img_size: int, settings:dict, key:int) -> Tuple[np.ndarray, np.ndarray, list]:
        
        
        image_noise = image_noise = settings[key]['image_noise']
        image_noise_weight = settings[key]['image_noise_weight']
        num_of_points = settings[key]['num_of_points']
        particle_weight=settings[key]['particle_weight']
        restrict2carbon = settings[key]['grid']['Restrict2carbon']
        restrict2Holes = settings[key]['grid']['Restrict2Holes']
        particle_noise=particle_noise = settings[key]['particle_noise']
        particle_noise_weight=settings[key]['particle_noise_weight']
        particle_size_min = settings[key]['particle_size_min']
        particle_size_max = settings[key]['particle_size_max']
        particle_size = np.random.randint(particle_size_min, particle_size_max)
        
        # generate voronoi array plus return voronoi points for each polygon
        vor_array, shad_arr, points = Grid().lacey_carbon_grid(inital_img_size) 

        graphene_carbon_location = vor_array.copy() # copy voronoi array
        graphene_carbon_location = cv2.resize(graphene_carbon_location[:int(inital_img_size*mag),:int(inital_img_size*mag)],
                                                            dsize=(inital_img_size,inital_img_size),
                                                            interpolation=cv2.INTER_NEAREST)

        graphene_carbon_location = 1 - graphene_carbon_location
    
        shad_arr = cv2.resize(shad_arr[:int(inital_img_size*mag),:int(inital_img_size*mag)],
                                                    dsize=(inital_img_size,inital_img_size),
                                                    interpolation=cv2.INTER_NEAREST)

        #plt.imshow(shad_arr, cmap='gray')
        #plt.axis('off')
        #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/paper2/bg_lacey3.png", shad_arr*255)
        #plt.show()

        if image_noise['type'] == 'perlin':
  
            noise_array = GeneratorUtils().perlin_noise(row = inital_img_size, # row size
                                            col = inital_img_size, # column size
                                            frequency = image_noise['frequency'], # frequency of perlin noise in the array
                                            octaves = image_noise['octaves'], # number of layers of noise 
                                            gain = image_noise['gain'], # how much eaxh layer is weighted against the previous layer
                                            lacunarity = image_noise['lacunarity']) # strength of the layer in question
            
        elif image_noise['type'] == 'simplexfracta':

            noise_array = GeneratorUtils().simplexfractal_noise(row = inital_img_size, # row size
                                                        col = inital_img_size, # column size
                                                        frequency=image_noise['frequency'], # frequency of the  noise in the array
                                                        octaves=image_noise['octaves'], # number of fractal noise layers.
                                                        gain=image_noise['gain'], 
                                                        lacunarity=image_noise['lacunarity'])# strenght of each layer.  

        else:
            noise_array = _create_zeros_like_array(shad_arr)

        lacey_carb = _add_arrays(shad_arr*5,  graphene_carbon_location*np.random.normal(0.6,0.1))
        lacey_carb = _add_arrays(lacey_carb, noise_array*image_noise_weight)

        if np.random.uniform(0,1) > 0.6:

            lacey_carb = GeneratorUtils().gradient_brightness(lacey_carb)

        lacey_carb = GeneratorUtils().normalise(lacey_carb)

        lacey_carb = lacey_carb - 0.1

        #plt.imshow(lacey_carb, cmap='gray')
        #plt.axis('off')
        #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/paper2/bg_lacey.png", lacey_carb*255)
        #plt.show()

        ###### particles

        points = np.empty([0, 2])

        for _ in range(num_of_points): # append random points to the list
            
            points = np.append(points,[[np.random.randint(0,inital_img_size),np.random.randint(0,inital_img_size)]], axis=0)

        if restrict2carbon:

            value_to_keep = 0
            points = Particles().remove_points_one_areas(graphene_carbon_location, particle_size, points, value_to_keep)

        if restrict2Holes:

            value_to_keep = 1
            points = Particles().remove_points_one_areas(graphene_carbon_location, particle_size, points, value_to_keep)
        
        ## generate the particle ellipse + it's associated mask
        ellipse_array, ellipse_array_mask, regions = Particles().create_particle(points=points, 
                                                                    mag=mag, 
                                                                    img_size=inital_img_size, 
                                                                    settings=settings,
                                                                    key=key)

        ellipse_array = GeneratorUtils().normalise(ellipse_array)

        particle_noise_array = _create_empty_like_array(ellipse_array_mask)
        

        if particle_noise['type'] == 'simplexfracta':

            simplexfracta_array = GeneratorUtils().simplexfractal_noise(row = inital_img_size, # row size
                                                        col = inital_img_size, # column size
                                                        frequency=particle_noise['frequency'], # frequency of the  noise in the array
                                                        octaves=particle_noise['octaves'], # number of fractal noise layers.
                                                        gain=particle_noise['gain'], 
                                                        lacunarity=particle_noise['lacunarity'])# strenght of each layer.  
                
            simplexfracta_array = cv2.resize(simplexfracta_array[:int(inital_img_size*mag),:int(inital_img_size*mag)],
                                                dsize=(inital_img_size,inital_img_size),
                                                interpolation=cv2.INTER_NEAREST)

            particle_noise_array = _create_empty_like_array(ellipse_array_mask)

            particle_noise_array[ellipse_array_mask == 1] = simplexfracta_array[ellipse_array_mask == 1]

        # Create a copy to avoid modifying the original background
        final_lacey_carb = lacey_carb.copy()
 
        final_lacey_carb = _add_arrays(lacey_carb, ellipse_array*particle_weight)

        final_lacey_carb = _add_arrays(final_lacey_carb, particle_noise_array*particle_noise_weight)

        final_lacey_carb = cv2.resize(final_lacey_carb,
                                dsize=(img_size,img_size),
                                interpolation=cv2.INTER_NEAREST)
        
        ellipse_array_mask = cv2.resize(ellipse_array_mask,
                                dsize=(img_size,img_size),
                                interpolation=cv2.INTER_NEAREST)
        
        scale = img_size/inital_img_size   

        regions = [[np.array(i)* scale for i in l] for l in regions]
        
        return final_lacey_carb, ellipse_array_mask, regions

    def create_holey_carbon(self, mag: np.ndarray, img_size: int, 
                            inital_img_size: int, settings:dict, key:int) -> Tuple[np.ndarray, np.ndarray, list]:

        image_noise = image_noise = settings[key]['image_noise']
        image_noise_weight = settings[key]['image_noise_weight']
        particle_size_min = settings[key]['particle_size_min']
        particle_size_max = settings[key]['particle_size_max']
        particle_size = np.random.randint(particle_size_min, particle_size_max)
        num_of_points = settings[key]['num_of_points']
        particle_weight=settings[key]['particle_weight']
        restrict2carbon = settings[key]['grid']['Restrict2carbon']
        restrict2Holes = settings[key]['grid']['Restrict2Holes']
        particle_noise=particle_noise = settings[key]['particle_noise']
        particle_noise_weight=settings[key]['particle_noise_weight']
        
        vor_array, shad_arr = Grid().holey_carbon_grid(inital_img_size) # generate voronoi array plus return voronoi points for each polygon

        graphene_carbon_location = vor_array.copy() # copy voronoi array

        graphene_carbon_location_binary = vor_array.copy()

        graphene_carbon_location_binary = cv2.resize(graphene_carbon_location_binary[:int(inital_img_size*mag),:int(inital_img_size*mag)],
                                                            dsize=(inital_img_size,inital_img_size),
                                                            interpolation=cv2.INTER_NEAREST)
        
        graphene_carbon_location = cv2.resize(graphene_carbon_location[:int(inital_img_size*mag),:int(inital_img_size*mag)],
                                                            dsize=(inital_img_size,inital_img_size),
                                                            interpolation=cv2.INTER_NEAREST)
    
        # contour, the smaller, the darker
        graphene_carbon_location[graphene_carbon_location==2] = np.random.normal(0.5, 0.1, 
                                                        len(graphene_carbon_location[graphene_carbon_location==2]))
        # background       
        graphene_carbon_location[graphene_carbon_location==1] = np.random.normal(0.6, 0.1,
                                                        len(graphene_carbon_location[graphene_carbon_location==1]))
        # carbon holes
        graphene_carbon_location[graphene_carbon_location==0] = np.random.normal(0.6, 0.1, 
                                                        len(graphene_carbon_location[graphene_carbon_location==0]))

        #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/Thesis/figures/coloured_bg_holey_ternary_img.png", GeneratorUtils().normalise(graphene_carbon_location)*255)

        shad_arr = cv2.resize(shad_arr[:int(inital_img_size*mag),:int(inital_img_size*mag)],
                                                    dsize=(inital_img_size,inital_img_size),
                                                    interpolation=cv2.INTER_NEAREST)
        
        #plt.imshow(shad_arr, cmap='gray')
        #plt.axis('off')
        #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/paper2/bg_holey_shad_array.png", shad_arr*255)
        #plt.show()

        if image_noise['type'] == 'perlin':

            noise_array = GeneratorUtils().perlin_noise(row = inital_img_size, # row size
                                            col = inital_img_size, # column size
                                            frequency = image_noise['frequency'], # frequency of perlin noise in the array
                                            octaves = image_noise['octaves'], # number of layers of noise 
                                            gain = image_noise['gain'], # how much eaxh layer is weighted against the previous layer
                                            lacunarity = image_noise['lacunarity']) # strength of the layer in question
            
            
        elif image_noise['type'] == 'simplexfracta':

            noise_array = GeneratorUtils().simplexfractal_noise(row = inital_img_size, # row size
                                                        col = inital_img_size, # column size
                                                        frequency=image_noise['frequency'], # frequency of the  noise in the array
                                                        octaves=image_noise['octaves'], # number of fractal noise layers.
                                                        gain=image_noise['gain'], 
                                                        lacunarity=image_noise['lacunarity'])# strenght of each layer.  

        else:
            noise_array = _create_zeros_like_array(shad_arr)

        #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/Thesis/figures/noise_array.png", GeneratorUtils().normalise(noise_array)*255)

        # add image, contours and noise
        holey_carb = _add_arrays(shad_arr*2,  graphene_carbon_location*2)
        holey_carb = _add_arrays(holey_carb, noise_array*image_noise_weight)
          
        holey_carb = GeneratorUtils().normalise(holey_carb)

        holey_carb = holey_carb - 0.1

        #plt.imshow(holey_carb, cmap='gray')
        #plt.axis('off')
        #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/paper2/bg_holey.png", holey_carb*255)
        #plt.show()
        #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/Thesis/figures/final_bg_holey.png", GeneratorUtils().normalise(holey_carb)*255)
        ###### particles

        points = np.empty([0, 2])

        for i in range(num_of_points): # append random points to the list
            
            points = np.append(points,[[np.random.randint(0,inital_img_size),np.random.randint(0,inital_img_size)]], axis=0)

        #points = np.array(points) # turn list into numpy array
        
        #points = ((points/inital_img_size)*(int(inital_img_size*mag))).astype('int') # re-normalise remaining points

        if restrict2carbon:

            value_to_keep = 1
            points = Particles().remove_points_one_areas(graphene_carbon_location_binary, particle_size, points, value_to_keep)

        if restrict2Holes:

            value_to_keep = 0
            points = Particles().remove_points_one_areas(graphene_carbon_location_binary, particle_size, points, value_to_keep)
        
        ## generate the particle ellipse + it's associated mask
        ellipse_array, ellipse_array_mask, regions = Particles().create_particle(points=points, mag=mag, img_size=inital_img_size, settings=settings, key=key)
        
        ellipse_array = GeneratorUtils().normalise(ellipse_array)

        particle_noise_array = _create_empty_like_array(ellipse_array_mask)

        if particle_noise['type'] == 'simplexfracta':

            simplexfracta_array = GeneratorUtils().simplexfractal_noise(row = inital_img_size, # row size
                                                        col = inital_img_size, # column size
                                                        frequency=particle_noise['frequency'], # frequency of the  noise in the array
                                                        octaves=particle_noise['octaves'], # number of fractal noise layers.
                                                        gain=particle_noise['gain'], 
                                                        lacunarity=particle_noise['lacunarity'])# strenght of each layer.  
 
                
            simplexfracta_array = cv2.resize(simplexfracta_array[:int(inital_img_size*mag),:int(inital_img_size*mag)],
                                                dsize=(inital_img_size,inital_img_size),
                                                interpolation=cv2.INTER_NEAREST)

            particle_noise_array = _create_empty_like_array(ellipse_array_mask)

            particle_noise_array[ellipse_array_mask == 1] = simplexfracta_array[ellipse_array_mask == 1]

        final_holey_carb = _add_arrays(holey_carb, ellipse_array*particle_weight)
        final_holey_carb = _add_arrays(final_holey_carb, particle_noise_array*particle_noise_weight)

        final_holey_carb = cv2.resize(final_holey_carb,
                                dsize=(img_size,img_size),
                                interpolation=cv2.INTER_NEAREST)
        
        ellipse_array_mask = cv2.resize(ellipse_array_mask,
                                dsize=(img_size,img_size),
                                interpolation=cv2.INTER_NEAREST)
        
        scale = img_size/inital_img_size   
            
        regions = [[np.array(i)* scale for i in l] for l in regions]

        return final_holey_carb, ellipse_array_mask, regions

    def create_continuous_carbon(self, inital_img_size:int, 
                                mag:np.ndarray, img_size:int, settings:dict, key:int) -> Tuple[np.ndarray, np.ndarray, list]:
        
        image_noise = image_noise = settings[key]['image_noise']
        image_noise_weight = settings[key]['image_noise_weight']
        num_of_points = settings[key]['num_of_points']
        particle_weight=settings[key]['particle_weight']
        particle_noise=particle_noise = settings[key]['particle_noise']
        particle_noise_weight=settings[key]['particle_noise_weight']
        
        if image_noise['type'] == 'perlin':

            noise_array = GeneratorUtils().perlin_noise(row = inital_img_size, # row size
                                            col = inital_img_size, # column size
                                            frequency = image_noise['frequency'], # frequency of perlin noise in the array
                                            octaves = image_noise['octaves'], # number of layers of noise 
                                            gain = image_noise['gain'], # how much eaxh layer is weighted against the previous layer
                                            lacunarity = image_noise['lacunarity']) # strength of the layer in question
            
            
        elif image_noise['type'] == 'simplexfracta':

            noise_array = GeneratorUtils().simplexfractal_noise(row = inital_img_size, # row size
                                                        col = inital_img_size, # column size
                                                        frequency=image_noise['frequency'], # frequency of the  noise in the array
                                                        octaves=image_noise['octaves'], # number of fractal noise layers.
                                                        gain=image_noise['gain'], 
                                                        lacunarity=image_noise['lacunarity'])# strenght of each layer.  


        else:
            noise_array = _create_zeros_array((inital_img_size, inital_img_size), np.float32)
            
        continuous_carb = noise_array.copy()*image_noise_weight
            
        points = np.empty([0, 2])
        
        for i in range(num_of_points): # append random points to the list
            
            points = np.append(points,[[np.random.randint(0,inital_img_size),np.random.randint(0,inital_img_size)]], axis=0)

        
        points = ((points/inital_img_size)*(int(inital_img_size*mag))).astype('int') # re-normalise remaining points

        ellipse_array, ellipse_array_mask, regions = Particles().create_particle(points=points, mag=mag, img_size=inital_img_size, settings=settings, key=key)
        
        ellipse_array = GeneratorUtils().normalise(ellipse_array)

        ellipse_array = cv2.resize(ellipse_array[:int(inital_img_size*mag),:int(inital_img_size*mag)],
                                                dsize=(inital_img_size,inital_img_size),
                                                interpolation=cv2.INTER_NEAREST)
        
        ## cut out section of voronoi array according to the magnification we want, resize back to inital_img_size x inital_img_size
        ellipse_array_mask = cv2.resize(ellipse_array_mask[:int(inital_img_size*mag),:int(inital_img_size*mag)],
                                                dsize=(inital_img_size,inital_img_size),
                                                interpolation=cv2.INTER_NEAREST)
        
        particle_noise_array = _create_empty_like_array(ellipse_array_mask)
    
        if particle_noise['type'] == 'simplexfracta':

            simplexfracta_array = GeneratorUtils().simplexfractal_noise(row = inital_img_size, # row size
                                                        col = inital_img_size, # column size
                                                        frequency=particle_noise['frequency'], # frequency of the  noise in the array
                                                        octaves=particle_noise['octaves'], # number of fractal noise layers.
                                                        gain=particle_noise['gain'], 
                                                        lacunarity=particle_noise['lacunarity'])# strenght of each layer.  
  
                
            simplexfracta_array = cv2.resize(simplexfracta_array[:int(inital_img_size*mag),:int(inital_img_size*mag)],
                                                dsize=(inital_img_size,inital_img_size),
                                                interpolation=cv2.INTER_NEAREST)

            particle_noise_array = _create_empty_like_array(ellipse_array_mask)

            particle_noise_array[ellipse_array_mask == 1] = simplexfracta_array[ellipse_array_mask == 1]

        final_continuous_carb = _add_arrays(continuous_carb, ellipse_array*particle_weight)
        final_continuous_carb = _add_arrays(final_continuous_carb, particle_noise_array*particle_noise_weight)
        
        #plt.imshow(continuous_carb, cmap='gray')
        #plt.axis('off')
        #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/paper2/bg_continuous2.png", continuous_carb*255)
        #plt.show()

        # keep only particles not being cutted by the resize
        regions = [item for item in regions if item[0] < int(inital_img_size*mag) and item[1] < int(inital_img_size*mag)]

        scale = inital_img_size/int(inital_img_size*mag)  


        regions = [[np.array(i)* scale for i in l] for l in regions] 
    
        final_continuous_carb = cv2.resize(final_continuous_carb,
                                dsize=(img_size,img_size),
                                interpolation=cv2.INTER_NEAREST)
        
        
        
        ellipse_array_mask = cv2.resize(ellipse_array_mask,
                                dsize=(img_size,img_size),
                                interpolation=cv2.INTER_NEAREST)
        
        scale = img_size/inital_img_size   
        
        regions = [[np.array(i)* scale for i in l] for l in regions]
            
        return final_continuous_carb, ellipse_array_mask, regions

    def create_exp_bg(self, im: np.ndarray, img_size: int,
                    settings:dict, key:int) -> Tuple[np.ndarray, np.ndarray, list]:
  
        particle_size_min = settings[key]['particle_size_min']
        particle_size_max = settings[key]['particle_size_max']
        particle_size = np.random.randint(particle_size_min, particle_size_max)
        particle_type = settings[key]['particle']
        num_of_points = settings[key]['num_of_points']
        particle_weight = settings[key]['particle_weight']
        particle_noise = particle_noise = settings[key]['particle_noise']
        particle_noise_weight = settings[key]['particle_noise_weight']
        
        #plt.imshow(im, cmap='gray')
        #plt.axis('off')
        #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/paper2/bg_experimental.png", im*255)
        #plt.show()

        ## generate the particle ellipse + it's associated mask
        points = np.empty([0, 2])

        for _ in range(num_of_points): # append random points to the list
            
            points = np.append(points,[[np.random.randint(0,img_size),np.random.randint(0,img_size)]], axis=0)

        points = ((points/img_size)*(int(img_size))).astype('int') # re-normalise remaining points


        ellipse_array, ellipse_array_mask, regions = Particles().create_particle(points=points, 
                                                            mag=1, img_size=img_size, settings=settings, key=key)
        
        ellipse_array = GeneratorUtils().normalise(ellipse_array)

        particle_noise_array = _create_empty_like_array(ellipse_array_mask)

        if particle_noise['type'] == 'simplexfracta':

            simplexfracta_array = GeneratorUtils().simplexfractal_noise(row = img_size, # row size
                                                        col = img_size, # column size
                                                        frequency=particle_noise['frequency'], # frequency of the  noise in the array
                                                        octaves=particle_noise['octaves'], # number of fractal noise layers.
                                                        gain=particle_noise['gain'], 
                                                        lacunarity=particle_noise['lacunarity'])# strenght of each layer.  

                
            # simplexfracta_array = cv2.resize(simplexfracta_array[:int(img_size),:int(img_size)],
            #                                     dsize=(img_size,img_size),
            #                                     interpolation=cv2.INTER_NEAREST)
            
            particle_noise_array = _create_empty_like_array(ellipse_array_mask)

            particle_noise_array[ellipse_array_mask == 1] = simplexfracta_array[ellipse_array_mask == 1]

        

        final_exp_bg = _add_arrays(im, ellipse_array*particle_weight)
        final_exp_bg = _add_arrays(final_exp_bg, particle_noise_array*particle_noise_weight)
        
        return final_exp_bg, ellipse_array_mask, regions

    def generate_synthetic_data(self, number_of_images: int, settings: dict, 
                                count:int, folder=None, data_train=None, data_val=None, 
                                data_train_coco=None, data_val_coco=None):

        """Generate Data as specified.
        
        Arguments:
            
            "number_of_images": 
                
                number of iterations through the program. The number of images
                generated is number*no_of_magnifications
                
            "folder":
                
                specify whether "training data" or "validation data".
                
            "min_uniform":
                
                minimum value for changing array brightness at the end.
                
            "max_uniform":
                    
                maximum value for changing array brightness at the end.
                
        Returns:
            
            Nothing. Just saves images and masks to file.
            
            """

        count_imgs_progress_bar = 0

        # Track class name to add unique classes to "categories" in COCO dataset
        class_name_unique_train = []
        class_name_unique_val = []
        
        for key in settings:
            
            particle_type = settings[key]['particle']
            grid_type = settings[key]['grid']['type']
            initial_i_min = settings[key]['initial_i_min']
            initial_i_max = settings[key]['initial_i_max']
            no_of_magnifications = settings[key]['no_of_magnifications']
            train_val_prob = settings[key]['train_val_ratio']
            class_name = settings[key]['class_name']           
            img_size = settings[key]['img_size']

            # Get all available filenames for experimental background
            if grid_type == 'experimental background':

                stepSize = settings[key]['grid']['stepSize'][0]
                windowSize = settings[key]['grid']['windowSize'][0]
                grid_path = settings[key]['grid']['path']

                # Get the list of image files
                image_files = []

                for subdir, dirs, files in os.walk(grid_path):

                    for item in files:

                        fullpath = os.path.join(grid_path, subdir, item)
                        image_files.append(fullpath)

            for count_imgs in range(number_of_images): # iterate through number specified by variable 'number'

                self.progress.emit(count_imgs_progress_bar)

                count_imgs_progress_bar += 1

                if np.random.uniform(0,1) < train_val_prob:

                    train_val = 'train'
                else:
                    train_val = 'val'

                # Populate categories information 

                if settings[key]['save'] == True:
                    if class_name not in class_name_unique_train and train_val == 'train':
                        
                        data_train_coco["categories"].append({"supercategory": class_name,
                                                            "id": int(key),
                                                            "name": class_name})
                        class_name_unique_train.append(class_name)
                        

                    if class_name not in class_name_unique_val and train_val == 'val':
                            
                        data_val_coco["categories"].append({"supercategory": class_name,
                                                            "id": int(key),
                                                            "name": class_name})
                        class_name_unique_val.append(class_name)

                initial_i = np.random.randint(initial_i_min, initial_i_max)
            
                gt_image=None
            
                for i in range(no_of_magnifications): # number of image magnifications per iteration
                            
                    i += initial_i
            
                    mag = np.round(1-(i/16),1) # define magnification for this image based on 'i' value
                    inital_img_size = 2048
                    
                    rescale_intensity = settings[key]['rescale_intensity']
                    rescale_intensity_prob = settings[key]['rescale_intensity_prob']
                    
                    # scale = inital_img_size/int(inital_img_size*mag)  
                    # particle_size = int(particle_size/scale)


                    # ----- LACEY CARBON GRID -----

                    ##### grid
                    if grid_type == 'lacey carbon':
                        
                        final_lacey_carb, ellipse_array_mask, regions = self.create_lacey_carbon(mag=mag, img_size=img_size, 
                                                                                            inital_img_size=inital_img_size,
                                                                                            settings=settings, key=key)
                        
                        final_regions = GetAnnotations().get_annotations(regions, particle_type, class_name)
                        
                        if settings[key]['save'] == True:
                            # loop over every image
                            if train_val == 'train':
                                data_train[str(count)+'.png'] = dict(
                                                        fileref='',
                                                        size=img_size*img_size,
                                                        filename=str(count)+'.png',
                                                        base64_img_data='',
                                                        file_attributes={},
                                                        regions=final_regions
                                                        )
                                
                                data_train_coco["images"].append({
                                                            "id": count,
                                                            "width": img_size,
                                                            "height": img_size,
                                                            "file_name": str(count)+'.png'
                                                        })
                                
                                data_train_coco = GetAnnotations().get_coco_annotations(regions, particle_type, count, data_train_coco, class_name)
                                            
                            if train_val == 'val':
                                data_val[str(count)+'.png'] = dict(
                                                        fileref='',
                                                        size=img_size*img_size,
                                                        filename=str(count)+'.png',
                                                        base64_img_data='',
                                                        file_attributes={},
                                                        regions=final_regions
                                                        )
                                
                                data_val_coco["images"].append({
                                                            "id": count,
                                                            "width": img_size,
                                                            "height": img_size,
                                                            "file_name": str(count)+'.png'
                                                        })
                                
                                data_val_coco = GetAnnotations().get_coco_annotations(regions, particle_type, count, data_val_coco, class_name)
                            
                        
                        
                        ellipse_array_mask = np.where(ellipse_array_mask>=1, 1., 0.)
                        #ellipse_array_mask = 1 - ellipse_array_mask
                        

                        final_lacey_carb = GeneratorUtils().normalise(final_lacey_carb)

                        final_lacey_carb = final_lacey_carb*255

                        final_lacey_carb = final_lacey_carb.astype(np.uint8)
                        #final_lacey_carb = np.clip(final_lacey_carb, 0, 1)

                        if (rescale_intensity == True) and (np.random.uniform(0,1) < rescale_intensity_prob):
                            p1, p99 = np.percentile(final_lacey_carb, (1, 99))
                            final_lacey_carb = exposure.rescale_intensity(final_lacey_carb, in_range=(p1, p99))

                        #final_lacey_carb = GeneratorUtils().normalise(final_lacey_carb)
                        if settings[key]['brighten'] == True:

                            brighten_rate = settings[key]['brighten_rate']
                                    
                            final_lacey_carb = np.clip(final_lacey_carb * brighten_rate, 0, 255).astype(np.uint8) 

                        if settings[key]['invert_image'] == True:

                            final_lacey_carb = 1 - final_lacey_carb

                        if settings[key]['save'] == True:

                            # final_lacey_carb = GeneratorUtils().normalise(final_lacey_carb)

                            # final_lacey_carb = final_lacey_carb*255

                            # final_lacey_carb = final_lacey_carb.astype(np.uint8)

                            cv2.imwrite(folder + '/' + train_val + '/' + str(count) + '.png', final_lacey_carb) # save image
                        # #cv2.imwrite(folder+'labels/'+str(count)+'.png',255*ellipse_array_mask) # save image

                            if settings[key]['binary_mask'] == True:
                            
                                cv2.imwrite(folder + '/' + train_val + '/binary_masks/' + str(count) + '.png', ellipse_array_mask*255) # save imageellipse_array_mask
                        
                        count = count + 1 # increment count for naming the next image

                        if settings[key]['plot_mask'] == True:
                            gt_image = self.plot_mask(final_lacey_carb, final_regions)

                        final_image = (final_lacey_carb).copy()

                    # ----- CONTINUOUS CARBON GRID -----
                    if grid_type == 'continuous carbon':
                        final_continuous_carb, ellipse_array_mask, regions = self.create_continuous_carbon(inital_img_size = inital_img_size, 
                                                                                                    mag = mag, img_size = img_size, 
                                                                                                    settings = settings, key = key)
                        
                        final_regions = GetAnnotations().get_annotations(regions, particle_type, class_name)
                        
                        if settings[key]['save'] == True:
                            # loop over every image
                            if train_val == 'train':
                                data_train[str(count)+'.png'] = dict(
                                                        fileref='',
                                                        size=img_size*img_size,
                                                        filename=str(count)+'.png',
                                                        base64_img_data='',
                                                        file_attributes={},
                                                        regions=final_regions
                                                        )
                                
                                data_train_coco["images"].append({
                                                            "id": count,
                                                            "width": img_size,
                                                            "height": img_size,
                                                            "file_name": str(count)+'.png'
                                                        })
                                
                                data_train_coco = GetAnnotations().get_coco_annotations(regions, particle_type, count, data_train_coco, class_name)
                                                
                            if train_val == 'val':
                                data_val[str(count)+'.png'] = dict(
                                                        fileref='',
                                                        size=img_size*img_size,
                                                        filename=str(count)+'.png',
                                                        base64_img_data='',
                                                        file_attributes={},
                                                        regions=final_regions
                                                        )
                                
                                data_val_coco["images"].append({
                                                            "id": count,
                                                            "width": img_size,
                                                            "height": img_size,
                                                            "file_name": str(count)+'.png'
                                                        })
                                
                                data_val_coco = GetAnnotations().get_coco_annotations(regions, particle_type, count, data_val_coco, class_name)
                                
                        ellipse_array_mask = np.where(ellipse_array_mask>=1, 1., 0.)
                        
                        #ellipse_array_mask = 1 - ellipse_array_mask
                        
                        final_continuous_carb = GeneratorUtils().normalise(final_continuous_carb)

                        final_continuous_carb = final_continuous_carb*255
                        final_continuous_carb = final_continuous_carb.astype(np.uint8) 

                        if (rescale_intensity == True) and (np.random.uniform(0,1) < rescale_intensity_prob):
                            p2, p98 = np.percentile(final_continuous_carb, (1, 99))
                            final_continuous_carb = exposure.rescale_intensity(final_continuous_carb, in_range=(p2, p98))
                        
                        #final_continuous_carb = GeneratorUtils().normalise(final_continuous_carb)

                        
                        if settings[key]['invert_image'] == True:

                            final_continuous_carb = 1 - final_continuous_carb
                        
                        if settings[key]['brighten'] == True:

                            brighten_rate = settings[key]['brighten_rate']
                                    
                            final_continuous_carb = np.clip(final_continuous_carb * brighten_rate, 0, 255).astype(np.uint8) 

                        
                        if settings[key]['save'] == True:

                            #final_continuous_carb = GeneratorUtils().normalise(final_continuous_carb)

                            #final_continuous_carb = final_continuous_carb*255

                            cv2.imwrite(folder + '/' + train_val + '/' + str(count) + '.png', final_continuous_carb) # save image
                        # #cv2.imwrite(folder+'labels/'+str(count)+'.png',255*ellipse_array_mask) # save image

                            if settings[key]['binary_mask'] == True:
                            
                                cv2.imwrite(folder + '/' + train_val + '/binary_masks/' + str(count) + '.png', ellipse_array_mask*255) # save imageellipse_array_mask
                            
                        count = count + 1 # increment count for naming the next image

                        if settings[key]['plot_mask'] == True:
                            
                            gt_image = self.plot_mask(final_continuous_carb, final_regions)
                        
                        final_image = (final_continuous_carb).copy()
                        
                    # ----- HOLEY CARBON GRID -----

                    if grid_type == 'holey carbon':
                
                        final_holey_carb, ellipse_array_mask, regions = self.create_holey_carbon(mag=mag, img_size=img_size, 
                                                                                            inital_img_size=inital_img_size,
                                                                                            settings=settings, key=key)
                        
                        final_regions = GetAnnotations().get_annotations(regions, particle_type, class_name)
                        
                        if settings[key]['save'] == True:
                            # loop over every image
                            if train_val == 'train':
                                data_train[str(count)+'.png'] = dict(
                                                        fileref='',
                                                        size=img_size*img_size,
                                                        filename=str(count)+'.png',
                                                        base64_img_data='',
                                                        file_attributes={},
                                                        regions=final_regions
                                                        )
                                
                                data_train_coco["images"].append({
                                                        "id": count,
                                                        "width": img_size,
                                                        "height": img_size,
                                                        "file_name": str(count)+'.png'
                                                    })

                                data_train_coco = GetAnnotations().get_coco_annotations(regions, particle_type, count, data_train_coco, class_name)

                            if train_val == 'val':
                                data_val[str(count)+'.png'] = dict(
                                                        fileref='',
                                                        size=img_size*img_size,
                                                        filename=str(count)+'.png',
                                                        base64_img_data='',
                                                        file_attributes={},
                                                        regions=final_regions
                                                        )
                                
                                data_val_coco["images"].append({
                                                        "id": count,
                                                        "width": img_size,
                                                        "height": img_size,
                                                        "file_name": str(count)+'.png'
                                                    })

                                data_val_coco = GetAnnotations().get_coco_annotations(regions, particle_type, count, data_val_coco, class_name)

                        ellipse_array_mask = np.where(ellipse_array_mask>=1, 1., 0.)
                        
                        if (rescale_intensity == True) and (np.random.uniform(0,1) < rescale_intensity_prob):
                            p2, p98 = np.percentile(final_holey_carb, (1, 99))
                            final_holey_carb = exposure.rescale_intensity(final_holey_carb, in_range=(p2, p98))
                        
                        final_holey_carb = GeneratorUtils().normalise(final_holey_carb)

                        final_holey_carb = final_holey_carb*255

                        final_holey_carb = final_holey_carb.astype(np.uint8)

                        if settings[key]['brighten'] == True:

                            brighten_rate = settings[key]['brighten_rate']
                                    
                            final_holey_carb = np.clip(final_holey_carb * brighten_rate, 0, 255).astype(np.uint8) 
                        
                        #plt.imshow(final_holey_carb, cmap='gray')
                        #plt.axis('off')
                        #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/paper2/final_holeycarbon.png", final_holey_carb*255)
                        #plt.show()

                        if settings[key]['invert_image'] == True:

                            final_holey_carb = 1 - final_holey_carb

                        if settings[key]['save'] == True:

                            # final_holey_carb = GeneratorUtils().normalise(final_holey_carb)

                            # final_holey_carb = final_holey_carb*255

                            cv2.imwrite(folder + '/' + train_val + '/' + str(count)+'.png', final_holey_carb) # save image
                        # cv2.imwrite(folder+'masks/'+str(count)+'.png',255*ellipse_array_mask) # save image
                        
                            if settings[key]['binary_mask'] == True:
                            
                                cv2.imwrite(folder + '/' + train_val + '/binary_masks/' + str(count) + '.png', ellipse_array_mask*255) # save imageellipse_array_mask
                        

                        count = count + 1 # increment count for naming the next image

                        if settings[key]['plot_mask'] == True:
                            gt_image = self.plot_mask(final_holey_carb, final_regions)

                            #plt.imshow(gt_image, cmap='gray')
                            #plt.axis('off')
                            #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/paper2/final_holeycarbon_gt.png", gt_image*255)
                            #plt.show()
                    
                        final_image = (final_holey_carb).copy()


                    # ----- EXPERIMENTAL GRID -----

                    if grid_type == 'experimental background':

                        # Total number of available images
                        total_images = len(image_files)

                        # Calculate the index of the image to be loaded
                        index = count_imgs % total_images

                        bg_image_path = image_files[index]

                        if (bg_image_path.endswith(".dm4")) or (bg_image_path.endswith(".dm3")):
                        
                            readDm4 = ReadDm4()
                            img, resolution, unit = readDm4.get_image_from_dm4(bg_image_path)
                                    
                            img = cv2.resize(img,
                                            dsize=(img_size, img_size),
                                            interpolation=cv2.INTER_NEAREST)
                            

                        if (bg_image_path.endswith(".png") or bg_image_path.endswith(".tiff")):

                            #fullpath = os.path.join(grid_path,subdir,item)  

                            img = cv2.imread(bg_image_path)  
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                            img = cv2.resize(img,
                                            dsize=(img_size, img_size),
                                            interpolation=cv2.INTER_NEAREST)
                            

                            resolution = 0

                        img = GeneratorUtils().normalise(img)
                            
                        # Calculate the number of windows in each dimension
                        n_h = (img.shape[0] - windowSize[0]) // stepSize + 1
                        n_w = (img.shape[1] - windowSize[1]) // stepSize + 1

                        #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/Thesis/figures/expbackground.png", GeneratorUtils().normalise(img)*255)

                        for y in range(n_w):

                            for x in range(n_h):

                                y_start = y * stepSize
                                y_end = y_start + windowSize[1]
                                x_start = x * stepSize
                                x_end = x_start + windowSize[0]

                                im = img[y_start:y_end, x_start:x_end]

                                #im = img[y:y + windowSize[1], x:x + windowSize[0]]
                                im = cv2.resize(im,
                                                dsize=(img_size, img_size),
                                                interpolation=cv2.INTER_NEAREST)
                                
                                
                            
                                if np.random.uniform(0,1) > 0.5:  
                                    transform = A.Compose([
                                                        A.RandomRotate90(),
                                                        A.Transpose(),
                                                        #A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45),
                                                        A.Blur(),
                                                        #A.OpticalDistortion(distort_limit = (-0.9, 0.9),shift_limit=(-0.9, 0.9)),
                                                        A.GridDistortion(),
                                                        A.RandomBrightnessContrast(brightness_limit=(-0.5, 0.5), contrast_limit=(-0.5, 0.5)),
                                                    ])
                                    random.seed(42) 

                                    im = transform(image=im)['image'] 

                                #im = cv2.flip(im, -1)

                                #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/Thesis/figures/RandomBrightnessContrast_expbackground.png", GeneratorUtils().normalise(im)*255)

                                final_exp_bg, ellipse_array_mask, regions = self.create_exp_bg(im = im, img_size = img_size,
                                                                                            settings = settings,
                                                                                            key = key)
                                
                                final_regions = GetAnnotations().get_annotations(regions, particle_type, class_name)
                                
                                if settings[key]['save'] == True:
                                    # loop over every image
                                    if train_val == 'train':
                                        data_train[str(count)+'.png'] = dict(
                                                                fileref='',
                                                                size=img_size*img_size,
                                                                filename=str(count)+'.png',
                                                                base64_img_data='',
                                                                file_attributes={},
                                                                regions=final_regions
                                                                )
                                        
                                        data_train_coco["images"].append({
                                                                "id": count,
                                                                "width": img_size,
                                                                "height": img_size,
                                                                "file_name": str(count)+'.png'
                                                            })

                                        data_train_coco = GetAnnotations().get_coco_annotations(regions, particle_type, count, data_train_coco, class_name)

                                    if train_val == 'val':
                                        data_val[str(count)+'.png'] = dict(
                                                                fileref='',
                                                                size=img_size*img_size,
                                                                filename=str(count)+'.png',
                                                                base64_img_data='',
                                                                file_attributes={},
                                                                regions=final_regions
                                                                )
                                        
                                        data_val_coco["images"].append({
                                                                "id": count,
                                                                "width": img_size,
                                                                "height": img_size,
                                                                "file_name": str(count)+'.png'
                                                            })

                                        data_val_coco = GetAnnotations().get_coco_annotations(regions, particle_type, count, data_val_coco, class_name)

                                ellipse_array_mask = np.where(ellipse_array_mask>=1, 1., 0.)
                                
                                if (rescale_intensity == True) and (np.random.uniform(0,1) < rescale_intensity_prob):
                                    
                                    p2, p98 = np.percentile(final_exp_bg, (1, 99))
                                    final_exp_bg = exposure.rescale_intensity(final_exp_bg, in_range=(p2, p98))
                                
                                final_exp_bg = GeneratorUtils().normalise(final_exp_bg)
                                
                                final_exp_bg = final_exp_bg*255

                                final_exp_bg = final_exp_bg.astype(np.uint8)

                                if settings[key]['brighten'] == True:

                                    brighten_rate = settings[key]['brighten_rate']
                                    
                                    final_exp_bg = np.clip(final_exp_bg * brighten_rate, 0, 255).astype(np.uint8) 

                                if settings[key]['invert_image'] == True:

                                    final_exp_bg = 1 - final_exp_bg

                                if settings[key]['save'] == True:

                                    # final_exp_bg = GeneratorUtils().normalise(final_exp_bg)

                                    # final_exp_bg = final_exp_bg*255

                                    cv2.imwrite(folder + '/' + train_val + '/' + str(count)+'.png', final_exp_bg) # save image
                                # cv2.imwrite(folder+'masks/'+str(count)+'.png',255*ellipse_array_mask) # save image

                               
                                
                                    if settings[key]['binary_mask'] == True:
                            
                                        cv2.imwrite(folder + '/' + train_val + '/binary_masks/' + str(count) + '.png', ellipse_array_mask*255) # save imageellipse_array_mask
                                
                                count = count + 1 # increment count for naming the next image
                                
                                if settings[key]['plot_mask'] == True:
                                    
                                    gt_image = self.plot_mask(final_exp_bg, final_regions)
                            
                                final_image = (final_exp_bg).copy()

                            else: pass


        return data_train, data_val, data_train_coco, data_val_coco, count, final_image, gt_image
    