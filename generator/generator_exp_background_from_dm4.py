from PIL import Image
import os.path, sys
import cv2
import matplotlib.pyplot as plt
import imgaug
import pyfastnoisesimd as fns
import numpy as np
import random
import albumentations as A
from particles import create_particle
from utils import perlin_noise, simplexfractal_noise
from annotations import get_annotations, get_coco_annotations
import json
from utils import  normalise
from read_dm4 import get_image_from_dm4
import skimage
import colorsys
import datetime

# Root directory of the project


# train_val = 'train'
# path = os.path.join(ROOT_DIR, "For Natalia png/")
# train_val_path = os.path.join(ROOT_DIR, "For Natalia png/" + train_val + '/')


#.dm4 images

#train_val_path = os.path.join(ROOT_DIR, "HoleyCarbon/")


##dirs = os.listdir(train_val_path)
#print(dirs)
def apply_mask(image, mask, color, alpha=0.2):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def plot_mask(final_lacey_carb, particle_type, final_regions):
    final_lacey_carb_3d = np.dstack((final_lacey_carb, final_lacey_carb, final_lacey_carb))
    
    # if particle_type =='polymer':
        
    #     mask = np.zeros([final_lacey_carb.shape[0], final_lacey_carb.shape[1], len(final_regions)],
    #                     dtype=np.uint8)

    #     for i, p in enumerate(final_regions):
    #         # Get indexes of pixels inside the polygon and set them to 1
    #     # rr, cc = skimage.draw.ellipse(np.round(p['cy']), np.round(p['cx']), np.round(p['ry']), np.round(p['rx']), rotation=np.deg2rad(np.round(p['theta'])))
    #         rr, cc = skimage.draw.ellipse(final_regions[p]['shape_attributes']['cx'], final_regions[p]['shape_attributes']['cy'], final_regions[p]['shape_attributes']['rx'], final_regions[p]['shape_attributes']['ry'], rotation=np.deg2rad(final_regions[p]['shape_attributes']['theta']))
    #         # rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
    #         unwnated_index_rr = np.where(np.logical_or(rr<0, rr>=2048))
    #         rr = np.delete(rr, unwnated_index_rr)
    #         cc = np.delete(cc, unwnated_index_rr)
    #         unwnated_index_cc = np.where(np.logical_or(cc<0, cc>=2048))
    #         rr = np.delete(rr, unwnated_index_cc)
    #         cc = np.delete(cc, unwnated_index_cc)
        
    #         mask[rr, cc, i] = 1
    # elif particle_type =='silica':

    mask = np.zeros([final_lacey_carb.shape[0], final_lacey_carb.shape[1], len(final_regions)],
                    dtype=np.uint8)

    for i, p in enumerate(final_regions):

        # Get indexes of pixels inside the polygon and set them to 1
        #rr, cc = skimage.draw.ellipse(np.round(p['cy']), np.round(p['cx']), np.round(p['ry']), np.round(p['rx']), rotation=np.deg2rad(np.round(p['theta'])))
        #rr, cc = skimage.draw.ellipse(np.round(p['cx']), np.round(p['cy']), np.round(p['rx']), np.round(p['ry']), rotation=-np.deg2rad(np.round(p['theta'])))
        rr, cc = skimage.draw.polygon(final_regions[p]['shape_attributes']['all_points_y'], final_regions[p]['shape_attributes']['all_points_x'])
        unwnated_index_rr = np.where(np.logical_or(rr<0, rr>=2048))
        rr = np.delete(rr, unwnated_index_rr)
        cc = np.delete(cc, unwnated_index_rr)
        unwnated_index_cc = np.where(np.logical_or(cc<0, cc>=2048))
        rr = np.delete(rr, unwnated_index_cc)
        cc = np.delete(cc, unwnated_index_cc)

        mask[rr, cc, i] = 1    

    hsv = [(i / mask.shape[2], 1, 1.0) for i in range(mask.shape[2])]
    color = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    
    for i in range(mask.shape[2]):

        image = apply_mask(final_lacey_carb_3d, mask[:,:,i], color[i])
    
    return image

def add_particles(img, height, resolution, data, count):
    
    if np.random.uniform(0,1) > 0.5:
    
        particle_type = 'polymer'
    else:
    
        particle_type = 'silica'

    if particle_type == 'polymer':
        # if resolution > 1.4:
            # particle_size = np.random.randint(140, 350)
            # num_of_points = 15
        # else:
            
            particle_size = np.random.randint(350, 1500)
            num_of_points = 2

    if particle_type == 'silica':
        # if resolution > 1.4:
            # particle_size = np.random.randint(40, 160)
            # num_of_points = 15
        # else:
            
            particle_size = np.random.randint(400, 600)
            num_of_points = 2

    ## generate the particle ellipse + it's associated mask
    points = np.empty([0, 2])

    for _ in range(num_of_points): # append random points to the list
        
        points = np.append(points,[[np.random.randint(0,height),np.random.randint(0,height)]], axis=0)

    
    points = ((points/height)*(int(height))).astype('int') # re-normalise remaining points
    

    ellipse_array, ellipse_array_mask, regions = create_particle(img,points=points, 
                                                                    mag=1, particle_size=particle_size, 
                                                                    img_size=2048, particle_type=particle_type)
    
    if particle_type == 'silica' and particle_size >=200:
        # add noise to the high mag particle
        simplexfracta_array = simplexfractal_noise( row = height, # row size
                                                    col = height, # column size
                                                    frequency=0.1, # frequency of the  noise in the array
                                                    octaves=15, # number of fractal noise layers.
                                                    gain=0.5, 
                                                    lacunarity=2.0 )# strenght of each layer.  
        
        simplexfracta_array = cv2.resize(simplexfracta_array[:int(height),:int(height)],
                                            dsize=(height,height),
                                            interpolation=cv2.INTER_NEAREST)
        
        simplexfracta_array = simplexfracta_array*2

        particle_noise = np.empty_like(ellipse_array_mask)

        particle_noise[ellipse_array_mask == 1] = simplexfracta_array[ellipse_array_mask == 1]

        particle_weight = 20
    else:
    
        particle_noise = np.empty_like(ellipse_array_mask)
        particle_weight = np.random.randint(13, 16)

    final = img + ellipse_array*particle_weight + particle_noise*10
    
    data_coco = get_coco_annotations(regions, particle_type, count, data)
        
    final_regions = get_annotations(regions, particle_type)
    
    return final, final_regions, particle_type, data_coco


def generate_experimental_background_data(folder, path, data_train, data_val, data_train_coco, data_val_coco, count):
    # data_train = {}
    # data_val = {}

    # data_train_coco = {'info':{
    #                         "description": 'Synthetic data with experimental background',
    #                         "year": '2024',
    #                         "contributor": "https://github.com/nataliaasaa",
    #                         "date_created": str(datetime.datetime.now())
    #                         },
    #                 'images':[],
    #                 'annotations':[],
    #                 'categories':[
    #                     {"supercategory": "particle",
    #                     "id": 1,
    #                     "name": "particle"}]
    #                 }
    # data_val_coco = {'info':{
    #                         "description": 'Synthetic data with experimental background',
    #                         "year": '2024',
    #                         "contributor": "https://github.com/nataliaasaa",
    #                         "date_created": str(datetime.datetime.now())
    #                         },
    #                 'images':[],
    #                 'annotations':[],
    #                 'categories':[
    #                     {"supercategory": "particle",
    #                     "id": 1,
    #                     "name": "particle"}]
    #                 }

    # count = 0

    for subdir, dirs, files in os.walk(path):
        
        for item in files:
            
            subdir_name = subdir.split('/')[-1]
            #split data in train/val 
            if np.random.uniform(0,1) > 0.3:

                train_val = 'train'
            else:
                train_val = 'val'

            if (item.endswith(".dm4")) or (item.endswith(".dm3")):
                
                fullpath = os.path.join(path, subdir,item)  

                im, resolution = get_image_from_dm4(fullpath)
          
            # print(resolution)

            if (item.endswith(".png")):
                fullpath = os.path.join(path,subdir,item)  

                im = cv2.imread(fullpath)  
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            
         
            # Remove images where part of the grid is showing, their resolution is > 0.029                  
            if resolution < 29.:

                height, width = im.shape
          
                file_number = 0
                
                stepSize = 768
                windowSize = (768, 768)
                
                count += 1 
                if train_val == 'train':
                    final, final_regions, particle_type, data_train_coco = add_particles(im, height=height, resolution=resolution, data = data_train_coco, count=count)
                
                if train_val == 'val':
                    final, final_regions, particle_type, data_val_coco = add_particles(im, height=height, resolution=resolution, data = data_val_coco, count=count)
                
                final = normalise(final)
                # image = plot_mask(final, particle_type, final_regions)
                
                # plt.imshow(image)
                # plt.axis('off')
                # plt.show()
                # plt.imshow(final*255,cmap='Greys_r')
                # plt.axis('off')
                # plt.show() 
                
                #print(ROOT_DIR + '/bg_' + train_val + '/' + subdir_name + '_' + file_name)
                file_name = item.split('.')[0]
                cv2.imwrite(folder + train_val + '/' + subdir_name + '_' + file_name + '.png', final*255) # save image
                
                if train_val == 'train':
                    data_train[subdir_name + '_' + file_name + '.png'] = dict(
                                    fileref='',
                                    size=height*height,
                                    filename= subdir_name + '_' + file_name + '.png',
                                    base64_img_data='',
                                    file_attributes={},
                                    regions=final_regions)
                    
                    data_train_coco["images"].append({
                                                "id": count,
                                                "width": height,
                                                "height": height,
                                                "file_name": subdir_name + '_' + file_name + '.png'
                                            })
                    
                if train_val == 'val':
                    data_val[subdir_name + '_' + file_name + '.png'] = dict(
                                    fileref='',
                                    size=height*height,
                                    filename= subdir_name + '_' + file_name + '.png',
                                    base64_img_data='',
                                    file_attributes={},
                                    regions=final_regions)
                    
                    data_val_coco["images"].append({
                                                "id": count,
                                                "width": height,
                                                "height": height,
                                                "file_name": subdir_name + '_' + file_name + '.png'
                                            })
                
                # for y in range(0, height, stepSize):

                #     for x in range(0, width, stepSize):

                #         img = im[y:y + windowSize[1], x:x + windowSize[0]]
                #         img = cv2.resize(img,
                #                         dsize=(1024,1024),
                #                         interpolation=cv2.INTER_NEAREST)
                        
                        
                    
                #         if np.random.uniform(0,1) > 0.5:  
                #             transform = A.Compose([
                                        
                #                                 A.RandomRotate90(),
                #                                 A.Transpose(),
                #                                 A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
                #                                 A.Blur(blur_limit=3),
                #                                 A.OpticalDistortion(),
                #                                 A.GridDistortion(),
                #                                 A.HueSaturationValue(),
                #                             ])
                #             random.seed(42) 

                #             img = transform(image=img)['image'] 
                        
                #         count += 1

                #         img = cv2.flip(img, -1)

                #         if train_val == 'train':
                #             final, final_regions, particle_type, data_train_coco = add_particles(im, height=height, resolution=resolution, data = data_train_coco, count=count)
                #         if train_val == 'val':
                #             final, final_regions, particle_type, data_val_coco = add_particles(im, height=height, resolution=resolution, data = data_val_coco, count=count)
                
                #         final = normalise(final)

                #         # image = plot_mask(final, particle_type, final_regions)
                #         # plt.imshow(image)
                #         # plt.axis('off')
                #         # plt.show()

                #         # plt.imshow(final*255,cmap='Greys_r')
                #         # plt.axis('off')
                #         # plt.show() 

                #         file_number += 1
                        
                #         # loop over every image
                #         if train_val == 'train':
                #             data_train[str(file_number) + '_' + subdir_name + '_' + file_name + '.png'] = dict(
                #                     fileref='',
                #                     size=height*height,
                #                     filename= str(file_number) + '_' + subdir_name + '_' + file_name + '.png',
                #                     base64_img_data='',
                #                     file_attributes={},
                #                     regions=final_regions)
                            
                #             data_train_coco["images"].append({
                #                                 "id": count,
                #                                 "width": height,
                #                                 "height": height,
                #                                 "file_name": str(file_number) + '_' + subdir_name + '_' + file_name + '.png'
                #                             })
                        
                #         if train_val == 'val':
                #             data_val[str(file_number) + '_' + subdir_name + '_' + file_name + '.png'] = dict(
                #                             fileref='',
                #                             size=height*height,
                #                             filename= str(file_number) + '_' + subdir_name + '_' + file_name + '.png',
                #                             base64_img_data='',
                #                             file_attributes={},
                #                             regions=final_regions)
                            
                #             data_val_coco["images"].append({
                #                                 "id": count,
                #                                 "width": height,
                #                                 "height": height,
                #                                 "file_name": str(file_number) + '_' + subdir_name + '_' + file_name + '.png'
                #                             })
                        
                #         cv2.imwrite(folder +  train_val + '/' + str(file_number) + '_' + subdir_name + '_' + file_name + '.png', final*255) # save image

    # train_json = os.path.join(folder + '/train/via_region_data.json')   
    # val_json = os.path.join(folder + '/val/via_region_data.json')   
    # train_coco = os.path.join(folder + '/train/instances_train.json')   
    # val_coco = os.path.join(folder + '/val/instances_val.json')   

    # with open(train_json, "w") as f:
    #         json.dump(data_train, f, ensure_ascii=False, indent=2)        

    # with open(val_json, "w") as f:
    #         json.dump(data_val, f, ensure_ascii=False, indent=2) 

    # with open(train_coco, "w") as f:
    #         json.dump(data_train_coco, f, ensure_ascii=False, indent=2)  

    # with open(val_coco, "w") as f:
    #         json.dump(data_val_coco, f, ensure_ascii=False, indent=2)    

    return data_train, data_val, data_train_coco, data_val_coco, count

# ROOT_DIR = os.path.abspath("./")
# path = os.path.join(ROOT_DIR, "all_bg/")
# folder = os.path.join(ROOT_DIR, "experimental_bg/")

# generate_experimental_background_data(folder, path)
