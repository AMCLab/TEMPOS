import numpy as np
import cv2
import math
from typing import Tuple
from .generator_utils import GeneratorUtils
from numba import jit
import matplotlib.pyplot as plt

@jit
def _create_zeros_array(size:tuple, dtype:np.dtype):

    zeros_array = np.zeros(size, dtype)

    return zeros_array

@jit
def _column_stack(x_coords:np.ndarray, y_coords:np.ndarray) -> np.ndarray:
    
    contour = np.column_stack((x_coords, y_coords)).astype(np.int32)

    return contour

@jit
def _add_arrays(array1:np.ndarray, array2:np.ndarray) -> np.ndarray:

    added_array = np.add(array1, array2)

    return added_array    

class Particles:

    def define_particle_shadow(self, array:np.ndarray, 
                               particle_type:str, kernel_size:int, sigmaX:int, illumination_type:str) -> np.ndarray:
        
        """
        Define 'shadow' region of the particle.
        
        Arguments:
            
            "array":  particle array to find shadows from
            
            "particle_type": type of particle being created

            "particle_size":  size of particle being created
            
            
        Returns:
            
            "blur": "coloured" particle array
            
            "white_shadow": stronger white shadow to mimic microscope ilumination
            
        """
    

        if illumination_type == 'no aberration':

            return np.zeros_like(array)

        else:
        
            size = array.shape[0] # change this size will change the size of the white shaddow in the particle. Smaller size, bigger and blurier shaddow
                
            np.random.seed() # random seed for rnadom number generators
            
            array = np.pad(array, ((3, 3), (3,3)), 'constant', constant_values=(0, 0))
            
            array = cv2.resize(array, (size,size), interpolation=cv2.INTER_NEAREST)
            
            ## vertical sobel filter array
            vertical_filter = cv2.Sobel(array, cv2.CV_32F, 1, 0)

            ## horizontal sobbel filter array
            horizontal_filter = cv2.Sobel(array, cv2.CV_32F, 0, 1)

            if illumination_type == 'full particle':
                
                orientation = np.sqrt(np.square(vertical_filter) + np.square(horizontal_filter))

            else:

                ## combine horizontal and vertical to orientation array
                orientation = np.arctan2(horizontal_filter,vertical_filter)*(180/np.pi)

                orientation = np.abs(orientation) # absolute value of orientation array.
            
            mask = _create_zeros_array((array.shape[0],array.shape[1]), dtype = np.float32) ## empty mask array
            
            mask[vertical_filter != 0] = 1 # set mask = 1 where vertical filter is not equal to 0
            
            mask[horizontal_filter != 0] = 1 # set mask = 1 where horiz filter is not equal to 0
            
            final_array = GeneratorUtils().normalise(orientation) # normalise array
            
            final_array = (final_array - 0.5)/0.5 # normalise between 1 and -1
            
            final_array[mask!=1] = 0 # set everything to zero that isnt 1 in mask in final array.
            
            blur = cv2.GaussianBlur(final_array, (kernel_size, kernel_size), sigmaX) # blur the image

            # white_shadow = blur.copy()
            
            # white_shadow[blur < 0] = 0

            # white_shadow = GeneratorUtils().normalise(white_shadow)

            # plt.imshow(white_shadow, cmap='gray')
            # plt.show()
    
            if illumination_type != 'full particle':

                blur = GeneratorUtils().normalise(blur) # normalise array

                blur = np.abs(((blur - 0.5)/0.5)) # normalise between 1 and -1

                blur = np.where(blur>0.5,1,0) # threshold + binarise

            else:

                blur[blur < 0] = 0

                blur = GeneratorUtils().normalise(blur)
            

            # plt.imshow(blur, cmap='gray')
            # plt.show()

            return blur

    def remove_list_elements(self, array:np.ndarray, particle_size:int, inital_img_size:int) -> np.ndarray:
        
        """
        remove points near the edge of the array.
        
        Arguments:
            
            "array":  array to be filtered.
            "particle_size":  size of the particle to be placed.
            "inital_img_size":  initial image size.
            
        Returns:
            
            "array":  filtered array.
            
            """
            
        index_1 = [coordinate[0] for coordinate in array] # x-coordintes for points
        
        ## list of indices to remove if index val > 2020 or < 25
        remove_index = [index for index, val in enumerate(index_1) if (val > inital_img_size - particle_size/2 or val < particle_size/2)]
        
        index_2 = [coordinate[1] for coordinate in array] # y-coordinates for points
        
        ## list of indices to remove if index val > 2020 or < 25
        remove_index = list(set(remove_index + [index for index, val in enumerate(index_2) if (val > inital_img_size - particle_size/2 or val < particle_size/2)]))
        
        array = np.delete(array,remove_index,axis=0) # delete the relevant indices
        
        return array

    def remove_points_outside_range(self, array: np.ndarray, lim:int) -> np.ndarray:
        
        """
        remove points outside new range after array is segmented based on magnification
        
        Arguments:
            
            "array":  array to be filtered.
            
            "lim": x and y coordinate limits based on magnification.
            
                    eg: if magnification = 0.5, then lim = 2048*0.5 = 1024.
            
        Returns:
            
            "array":  filtered array.
            
            """
            
        index_1 = [coordinate[0] for coordinate in array] # x-coordintes for points
        
        ## list of indices to remove if index val > lim - 30 or < 30
        remove_index = [index for index, val in enumerate(index_1) if (val > lim-30 or val < 30)]
        
        index_2 = [coordinate[1] for coordinate in array] # y-coordinates for points
        
        ## list of indices to remove if index val > lim - 30 or < 30
        remove_index = list(set(remove_index + [index for index, val in enumerate(index_2) if (val > lim-30 or val < 30)]))
        
        array = np.delete(array,remove_index,axis=0) # delete the relevant indices
        
        return array

    def remove_points_one_areas(self, array: np.ndarray, particle_size: int, points: np.ndarray, value_to_keep:int) -> np.ndarray:
        
        """
        remove points that fall on lacey carbon area. Silica particles generally fall
        in graphene areas.
        
        Arguments:
            
            "array":  lacey-carbon array.

            "particle_size":  size of the particle to be placed.
            
            "points": points where we could place particles.

            "value_to_keep": either 0 or 1, depending if particles will be placed inside or outside holes. 
            
            0 for particles ouside holes if background is lacey carbon;
            1 for particles inside holes if background is lacey carbon;

            1 for particles ouside holes if background is holey carbon;
            0 for particles inside holes if background is holey carbon;
            
        Returns:
            
            "array":  filtered array.
            
            """
            
        non_one_points = [] # points which meet criteria
        
        for point in points: # iterate through the points
            
            sub_section =  array[int(point[0]-particle_size/2):int(point[0]+particle_size/2),int(point[1]-particle_size/2):int(point[1]+particle_size/2)]
            
            ## if point correponds with value of zero in the array, then add to final list 
            if np.all(sub_section == value_to_keep):
            
                non_one_points.append(list(point)) # append point to list
            
        return np.array(non_one_points)

    def image_rotation(self, image, angle):
        
        """
        rotating the array containing the silica particle
        
        Arguments:
            
            "image":  image of silica particle
            
            "angle": angle of rotation
            
        Returns:
            
            "result":  rotated array
            
            """
            
        row,col = image.shape # assign row and column length values based on array dimensions
        
        n = (row/np.sqrt(2))*math.sin(angle*math.pi/180) # length of the side of the rotated section we want to cut out
        
        pts1 = np.float32([[n,0],[row,n],[0,col-n],[row-n,col]]) # coordinates that you want to Perspective Transform
        
        pts2 = np.float32([[0,0],[row,0],[0,col],[row,col]]) # Size of the Transformed Image (original array size)

        M = cv2.getPerspectiveTransform(pts1,pts2) # computer transformation matrix
        
        result = cv2.warpPerspective(image,M,(row,col)) # apply transformation matrix to the original image
        
        return result

    def gradient_brightness(self, array:np.ndarray) -> np.ndarray:
        
        """
        create gradient brightness array to change particle brightness
        
        Arguments:
            
            "array":  array to change the brightness of
            
        Returns:
            
            "array":  gradient brightness particle array
            
            """
        
        random = np.random.uniform(0,1) # generate random number.
        
        row,col = array.shape[0],array.shape[1] # assign values to row and col variables, which are the shape of the input array.

        brighten = np.random.uniform(0.3,0.4)       #for the hole image
            
        segment = np.arange(0,brighten,brighten/row,dtype='float32') # segment to be brightened
              
        if len(segment) > row:
            
            segment = segment[:row].reshape(row,1)
            
        else:
            
            segment = segment.reshape(row,1)
        
        segment = np.tile(segment,col) # turn segment (500,1) vector into repeating (500,500) verctor, so that we account for all array columns
        
        if random > 0.5: # if random number greater than 0.5, flip the array so dimming happens in opposite direction
            
            segment = np.flip(segment) # flip the array for blurring in the opposite direction
    
        array = np.multiply(segment,array) # decrease brightness by multiplying array and segment
        
        array = array/np.max(array) # normalise
        
        return array

    def create_particle(self, points:np.ndarray, mag:np.ndarray,
                        img_size:int, settings:dict, key:int) -> Tuple[np.ndarray, np.ndarray, list]:
        
        """
        create silica particle with gradient change in the brightness of the ellipse
        
        Arguments:
            
            "points":  potential where particle will be placed

            "mag":  magnification

            "particle_size": size of the particle
            
            "img_size": size of image
            
            "particle_type": type of particle to be created
            
              
        Returns:
            
            "img_array":  array containing particles
            
            "mask_array": corresponding mask array for the particles

            "tracking_particles": particles annotations
            
            """
        
        particle_size_min = settings[key]['particle_size_min']
        particle_size_max = settings[key]['particle_size_max']
        particle_size = np.random.randint(particle_size_min, particle_size_max)
        particle_type=settings[key]['particle']
        vertices_min=settings[key]['vertices_min']
        vertices_max=settings[key]['vertices_max']
        irregularity = settings[key]['irregularity']
        generate_cluster = settings[key]['generate_cluster']
        if generate_cluster == True:
            cluster_type = settings[key]['cluster_type']
            rows_min = settings[key]['num_rows_min']
            rows_max = settings[key]['num_rows_max']
            cols_min = settings[key]['num_cols_min']
            cols_max = settings[key]['num_cols_max']
            width_min = settings[key]['num_width_min']
            width_max = settings[key]['num_width_max']
            cluster_random_num = settings[key]['cluster_random_num']
        max_overlap = 1 - settings[key]['max_overlap']
        illumination_weight = settings[key]['illumination_weight']
        illumination_type = settings[key]['illumination_type']
        illumination_kernel_size = settings[key]['illumination_kernel']
        sigmaX = settings[key]['illumination_sigmaX']
        axes_ratio = settings[key]['axes_ratio']
        gradient_weight = settings[key]['gradient_weight']

        min_particle_darkness = -3
        max_particle_darkness = -2

        # if particle_type == 'fibrils':

        #     max_overlap = 0  # allow particle overlap

        # else:
        #     max_overlap = 0.6 # allow particle overlap less then 60%

        points = self.remove_list_elements(points, particle_size, img_size) # remove unwanted points

        size = 500 # size of the array
        
        img_array = _create_zeros_array((img_size,img_size), dtype = np.float32) # instantiate the image array
        
        mask_array = _create_zeros_array((img_size,img_size), dtype = np.float32) # instantiate the mask array

        tracking_particles = [[-10000,-10000, -10000, -10000, -10000]] # Iniciate with a small number so the first point won't fall into the if condition

        
        if len(points) == 0:
            
            points = [(int(img_size/2),int(img_size/2))]

        if len(points) == 1:

            point = [np.random.randint(0,int(img_size/2)),np.random.randint(0,int(img_size/2))]

            
        for point in points: # iterate through the points to plot
        
            tracking_list_x = list(list(zip(*tracking_particles))[0])
            tracking_list_y = list(list(zip(*tracking_particles))[1])

        
            if any(np.abs(point[0] - point_x)<particle_size*max_overlap for point_x in tracking_list_x) and any(np.abs(point[1] - point_y)<particle_size*max_overlap for point_y in tracking_list_y):

                pass
            
            else:    
                centre = (int(size/2),int(size/2)) # centre of particle array based on 'size' variable
                
                gradient_particle_array = _create_zeros_array((size,size), dtype = np.float32) # empty array for particle color gradient
                
                particle_array = _create_zeros_array((size,size), dtype = np.float32) # empty array for particle color
                
                ellipse_array = _create_zeros_array((size,size), dtype = np.float32) # empty array to draw particle circle in
                
                if particle_type == 'ellipse':
                    
                    axes_length = (int((np.random.randint(240,250)*axes_ratio)),np.random.randint(240,250)) # length of the major and minor axis of ellipse
            
                if particle_type == 'silica' or particle_type == 'gold' or particle_type == 'polygon':
                
                    axes_length = (int((np.random.randint(230,250)*axes_ratio)),np.random.randint(230,250)) # length of the major and minor axis of ellipse
                
                if particle_type == 'fibrils':
                
                    axes_length = (np.random.randint(20,35),np.random.randint(300,500)) # length of the major and minor axis of ellipse

                if particle_type == 'rod':

                    axes_length = (np.random.randint(90, 120),np.random.randint(40,120))

                angle = np.random.randint(0,180) # angle ellipse rotates through

                start = 0 # proportion of ellipse fill in, start angle
                
                end = 360 # proportion of ellipse filled in, end angle
            
                if particle_type == 'silica' or particle_type == 'gold' or particle_type == 'polygon':
                    
                    num_vertices = np.random.randint(vertices_min, vertices_max)
                    # Add random rotation angle
                    rotation_angle = np.random.uniform(0, 2 * math.pi)
        
                    angles = np.linspace(0, 2 * math.pi, num_vertices, endpoint=False) + rotation_angle
                    
                    radius = int(np.random.randint(particle_size_min, particle_size_max)/2)
                

                    # Generate random side lengths with noise
                    side_lengths = [radius + np.random.uniform(-radius * irregularity, radius * irregularity) for _ in range(num_vertices)]
                    x_coords = [int(centre[0] + length * np.cos(angle)) for angle, length in zip(angles, side_lengths)]
                    y_coords = [int(centre[1] + length * np.sin(angle)) for angle, length in zip(angles, side_lengths)]
                    
                    x_coords = np.array(x_coords)
                    y_coords = np.array(y_coords)

                    # Create an array of points for the contour
                    contour = _column_stack(x_coords, y_coords)

                    # Draw the bumpy circle contour
                    cv2.polylines(ellipse_array, [contour], isClosed=True, color=(255, 255, 255), thickness=1)

                    # Fill the interior of the circle
                    ellipse_array = (cv2.fillPoly(ellipse_array, [contour], color=(255, 255, 255))/255).astype('uint8')
                
                    #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/Thesis/figures/"+ str(num_vertices) + "vert_"+ str(irregularity) +"irreg_particles_polygons.png", GeneratorUtils().normalise(ellipse_array)*255)
             
                elif particle_type == 'fibrils':
                
                    rectangle_points = (centre, (axes_length[0], axes_length[1]), angle)
                    rectangle_box = cv2.boxPoints(rectangle_points)
                    rectangle_box = np.intp(rectangle_box)
                    ellipse_array = (cv2.drawContours(ellipse_array,[rectangle_box],0, (255,255,255),-1)/255).astype('uint8')

                    #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/Thesis/figures/particles_fibrils.png", GeneratorUtils().normalise(ellipse_array)*255)
             
                    # Find mask contour using skimage find_contours
                    contours = GeneratorUtils().get_mask_contours(ellipse_array)
                    x_coords = [point[0] for contour in contours for point in contour]
                    y_coords = [point[1] for contour in contours for point in contour]

                elif particle_type == 'rod':

                    # create the rod with diferent directions
                    if np.random.uniform(0,1) > 0.5:

                        x_coords = [(centre[0] - axes_length[0]), centre[0] + axes_length[0]]
                        y_coords = [(centre[1] + axes_length[1]), centre[1] - axes_length[1]]
                        
                        start_point = (x_coords[0], y_coords[0])

                        end_point = (x_coords[1], y_coords[1])

                    else: 
                        
                        x_coords = [(centre[0] - axes_length[0]), centre[0] + axes_length[0]]
                        y_coords = [(centre[1] - axes_length[1]), centre[1] + axes_length[1]]
                        
                        start_point = (x_coords[0], y_coords[0])

                        end_point = (x_coords[1], y_coords[1])

                    thickness = np.random.randint(85, 205)   
                    ellipse_array = (cv2.line(ellipse_array, start_point, end_point, (255,255,255), thickness)/255).astype('uint8') 

                    #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/Thesis/figures/particles_rods.png", GeneratorUtils().normalise(ellipse_array)*255)
             

                    # Find mask contour using skimage find_contours
                    contours = GeneratorUtils().get_mask_contours(ellipse_array)
                    x_coords = [point[0] for contour in contours for point in contour]
                    y_coords = [point[1] for contour in contours for point in contour]

                else:

                    # draw circle in the array
                    #ellipse_array = (cv2.ellipse(ellipse_array,centre,axes_length,angle,start,end,(255,255,255),-1)/255).astype('uint8') 

                    cx, cy = centre
                    rx, ry = axes_length
                    theta = angle

                    angles = np.linspace(0, 2*np.pi, 200)

                    # Calculate x and y coordinates of points on the ellipse to save as polygon
                    x_coords = cx + ry * np.cos(angles) * np.cos(theta) - rx * np.sin(angles) * np.sin(theta)
                    y_coords = cy + ry * np.cos(angles) * np.sin(theta) + rx * np.sin(angles) * np.cos(theta)
                    
                    # Create an array of points for the contour
                    contour = _column_stack(x_coords, y_coords)

                    # Draw the circle contour
                    cv2.polylines(ellipse_array, [contour], isClosed=True, color=(255, 255, 255), thickness=1)

                    # Fill the interior of the circle
                    ellipse_array = (cv2.fillPoly(ellipse_array, [contour], color=(255, 255, 255))/255).astype('uint8')

                    #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/Thesis/figures/particles_ellipse.png", GeneratorUtils().normalise(ellipse_array)*255)
             
                    #plt.imshow(ellipse_array, cmap='gray')
                    #plt.axis('off')
                    #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/paper2/particle_ellipse0.png",ellipse_array*255)
                    #plt.show()

                noise_array = np.random.normal(0.5,0.2,(size,size)) # fill array with Gaussian noise

                noise_array = GeneratorUtils().normalise(noise_array) # normalise the array

                #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/Thesis/figures/particles_noise.png", GeneratorUtils().normalise(noise_array)*255)
             
                    
                gradient_array = self.gradient_brightness(noise_array) # create gradient change in brightness for array
                
                gradient_array = GeneratorUtils().normalise(gradient_array) # normalise the array

                #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/Thesis/figures/particles_gradient_array.png", GeneratorUtils().normalise(gradient_array)*255)
             

                #plt.imshow(gradient_array, cmap='gray')
                #plt.axis('off')
                #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/paper2/particle_ellipse1.png", gradient_array*255)
                #plt.show()
            
                gradient_particle_array[ellipse_array==1] = gradient_array[ellipse_array==1] # final gradient for particle

                #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/Thesis/figures/particles_gradient_particle_array.png", GeneratorUtils().normalise(gradient_particle_array)*255)
             

                #plt.imshow(gradient_particle_array, cmap='gray')
                #plt.axis('off')
                #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/paper2/particle_ellipse2.png", gradient_particle_array*255)
                #plt.show()

                particle_darkness = np.random.uniform(min_particle_darkness, max_particle_darkness) # particle color pixel values
                
                # coloring in the particle with random uniform noise
                particle_array[ellipse_array==1] = np.random.uniform(particle_darkness,particle_darkness+0.5,len(ellipse_array[ellipse_array==1]))

                #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/Thesis/figures/particles_particle_array.png", GeneratorUtils().normalise(particle_array)*255)
             

                # resize the particle array 
                final_particle_array = cv2.resize(particle_array+gradient_particle_array*gradient_weight,(particle_size,particle_size),interpolation=cv2.INTER_NEAREST)

                #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/Thesis/figures/particles_final_particle_array.png", GeneratorUtils().normalise(final_particle_array)*255)
             
                # if scale is defined as new_size/old_size, we multiply the axes_lenght by the scale
                scale = particle_size/particle_array.shape[0]  # re escale axis lenth for regions     
        
                axes_length = np.array(axes_length)*scale
            
                #if (particle_type =='fibrils' or particle_type =='silica' or particle_type =='polygon' or particle_type =='gold' or particle_type =='rod'):

                x_coords = np.array(x_coords)* scale
                y_coords = np.array(y_coords)* scale
                centre = np.array(centre)*scale
                    
                ## resize the particle mask array
                ellipse_array = cv2.resize(ellipse_array,(particle_size,particle_size),interpolation=cv2.INTER_NEAREST)
                
                # define particle shadow
                shadow_array = self.define_particle_shadow(ellipse_array, particle_type, illumination_kernel_size, sigmaX, illumination_type) 

                #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/Thesis/figures/particles_shadow_array.png", GeneratorUtils().normalise(shadow_array)*255)
             

                # white_shadow =  shadow_array.copy()

                shadow_array = cv2.resize(shadow_array, dsize=(particle_size, particle_size), interpolation=cv2.INTER_NEAREST)
                
                # white_shadow = cv2.resize(white_shadow, dsize=(particle_size, particle_size), interpolation=cv2.INTER_NEAREST)
                
                #plt.imshow(shadow_array, cmap='gray')
                #plt.axis('off')
                #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/paper2/particle_shadow_array.png", shadow_array*255)
                #plt.show()

                #plt.imshow(white_shadow, cmap='gray')
                #plt.axis('off')
                #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/paper2/particle_ellipse_white_shadow.png", white_shadow*255)
                #plt.show()

                # Pad array to place white shaddow outside particle
                final_particle_array = np.pad(final_particle_array, ((5, 5), (5, 5)), 'constant', constant_values=(0, 0))
            
                scale = particle_size/final_particle_array.shape[0]      # re escale axis lenth for regions     

                # Reescale annotations
                axes_length = np.array((axes_length) + 2.5) * scale

                final_particle_array = cv2.resize(final_particle_array, dsize=(particle_size, particle_size), interpolation=cv2.INTER_NEAREST)
                
                #plt.imshow(final_particle_array, cmap='gray')
                #plt.axis('off')
                #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/paper2/particle_ellipse3.png", final_particle_array*255)
                #plt.show()

                #if (particle_type =='fibrils' or particle_type =='silica' or particle_type =='polygon' or particle_type =='gold' or particle_type =='rod'):
                
                x_coords = (np.array(x_coords)-5)* scale
                y_coords = (np.array(y_coords)-5)* scale
                centre = (np.array(centre)-5)*scale

                final_particle_array[ellipse_array == 0] = 0
                
                # Need to pad the mask too so particle and mask have the same size 
                ellipse_array = np.pad(ellipse_array, (5, 5), 'constant', constant_values=(0, 0))
                    
                ellipse_array = cv2.resize(ellipse_array, dsize=(particle_size, particle_size), interpolation=cv2.INTER_NEAREST)
            
                if mag<0.3 and particle_type == 'ellipse':

                    # Add dark shaddow to the top of the particle
                    if np.random.uniform(0,1) > 0.7:
                        
                        number_for_size = np.random.randint(10, 15)
                    
                        number_for_position = np.random.randint(5, particle_size - number_for_size )
                        
                        surface_array = np.pad(final_particle_array, (number_for_position, number_for_position + number_for_size), 'constant', constant_values=(0, 0))

                        surface_array = cv2.resize(surface_array, dsize=(particle_size, particle_size), interpolation=cv2.INTER_NEAREST)

                        # Not too dark when the magnification is high
                        final_particle_array = final_particle_array.copy() + surface_array/3
                
                # if particle_type == 'fibrils' or particle_type == 'rod' or particle_type == 'polygon':

                #     white_shadow = 0
                
                # else:

                final_particle_array = final_particle_array.copy() + shadow_array*illumination_weight # add particle array to it's shadow array

                #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/Thesis/figures/particles_final_particle_array+" +str(illumination_type)+ "illumination.png", GeneratorUtils().normalise(final_particle_array)*255)

                ## place particle in the overall image array
                #prob = np.random.uniform(0,1)          

                x_coords = x_coords+(point[1] - centre[1])
                y_coords = y_coords+(point[0] - centre[0])
                
                r_x = 0
                r_y = 0

                if generate_cluster == True and cluster_type == 'random':
                    
                    # x_coords = x_coords+(point[1] - centre[1])
                    # y_coords = y_coords+(point[0] - centre[0])

                    for _ in range(0, cluster_random_num):
                        
                        if any(np.abs(point[0] + r_x - point_x) < particle_size*max_overlap for point_x in list(list(zip(*tracking_particles))[0])) and any(np.abs(point[1]+ r_y - point_y)<particle_size*max_overlap for point_y in list(list(zip(*tracking_particles))[1])):
                        
                            pass
                        
                    
                        else: 
                            
                            if point[0] + r_x > 0 and point[1]+ r_y>0:
                            
                                try:
                                    
                                    #particle_to_be_placed = _add_arrays(final_particle_array, white_shadow*illumination_weight)

                                    img_array = self.place_particles(img_array, final_particle_array,int(point[0])+r_x,int(point[1])+r_y)
                                    
                                    mask_array = self.place_particles(mask_array,ellipse_array,int(point[0])+r_x,int(point[1])+r_y)
                                
                                    x_coords_cluster = x_coords+(int(point[1])+r_y - point[1])
                                    y_coords_cluster = y_coords+(int(point[0])+r_x - point[0])
                                    
                                    tracking_particles.append([point[0] + r_x, point[1] + r_y, x_coords_cluster, y_coords_cluster])
                                    
                                except: pass
                
                        # r_x += np.random.choice([particle_size+np.random.randint(-5, 2), -particle_size-np.random.randint(-5, 2), 0+np.random.randint(-5, 2)])
                                
                        # r_y += np.random.choice([particle_size+np.random.randint(-5, 2), 0+np.random.randint(-5, 2)])
                        r_x += np.random.choice([particle_size + np.random.randint(0, 5),    # Move right (positive X)
                                                -particle_size - np.random.randint(0, 5),   # Move left (negative X)
                                                np.random.randint(0, 5)])  # Allow small X noise but prevent backtracking


                        r_y += np.random.choice([particle_size + np.random.randint(0, 5),    # Move down (positive Y)
                                                -particle_size - np.random.randint(0, 5),   # Move up (negative Y)
                                                np.random.randint(0, 5)])   # Allow small Y noise but prevent backtracking
                              
                #else:    

                    
                    #if particle_type =='fibrils' or particle_type == 'silica' or particle_type == 'gold' or particle_type == 'polygon' or particle_type == 'rod':
                        
                    # change scale of original particle centre (250,250) and move x and y coords to new position (new coords = old coords + (new centre - old centre))
                    # x_coords = x_coords+(point[1] - centre[1])
                    # y_coords = y_coords+(point[0] - centre[0])
                    
                    # For gold clusters
                elif (generate_cluster == True and cluster_type == 'honeycomb'):
                    
                    # When the number of potential particle is 1,
                    # Ensure cluster starting point is somewhere in the first quadrant of the image, 
                    # so the cluster is visible 
                    if len(points) == 1:

                        point2 = [np.random.randint(0,int(img_size/2)),np.random.randint(0,int(img_size/2))]
                        # reescale the coordinate position for the new point
                        x_coords = x_coords+(point2[1] - point[1])
                        y_coords = y_coords+(point2[0] - point[0])

                    else:
                        # change nothing but create variable point2
                        point2 = point.copy()

                    num_rows = np.random.randint(rows_min, rows_max)
                    num_cols = np.random.randint(cols_min, cols_max)

                    for row in range(num_rows):

                        for col in range(num_cols):

                            width_size = np.random.randint(width_min,width_max) + (axes_length[0])

                            if particle_type == 'rod':
                                height_size = np.random.randint(width_min,width_max) * (thickness*scale/2)
                            else:
                                height_size = np.random.randint(width_min,width_max) + (axes_length[1])


                            x = int(point2[0]) + col * width_size
                            y = int(point2[1]) + row * height_size
                            
                            if any(np.abs(x - point_x) < particle_size*max_overlap for point_x in list(list(zip(*tracking_particles))[0])) and any(np.abs(y - point_y)<particle_size*max_overlap for point_y in list(list(zip(*tracking_particles))[1])):
                
                                pass    

                            else:
                                if col % 2 == 1:

                                    y += height_size / 2

                                try:

                                    #particle_to_be_placed = _add_arrays(final_particle_array, white_shadow*illumination_weight)

                                    img_array = self.place_particles(img_array, final_particle_array, int(x),int(y))
                                            
                                    mask_array = self.place_particles(mask_array,ellipse_array,int(x),int(y))
                                        
                                    # change scale of original particle centre (250,250) and move x and y coords to new position (new coords = old coords + (new centre - old centre))
                                    x_coords_cluster = x_coords+(y - point2[1])
                                    y_coords_cluster = y_coords+(x - point2[0])

                                    tracking_particles.append([int(x), int(y), x_coords_cluster, y_coords_cluster])
                                
                                except: pass
                            
                    # elif (particle_type == 'rod' and generate_cluster == True):
                        
                    #     # For rod clusters
                    #     width_size = 5 * (axes_length[0]/2)
                    #     height_size = 5 * (thickness*scale/2)

                    #     num_rows = 3
                    #     num_cols = 3

                    #     for row in range(num_rows):

                    #         for col in range(num_cols):

                    #             x = int(point[0]) + col * width_size
                    #             y = int(point[1]) + row * height_size

                    #             if any(np.abs(x - point_x) < particle_size*max_overlap for point_x in list(list(zip(*tracking_particles))[0])) and any(np.abs(y - point_y)<particle_size*max_overlap for point_y in list(list(zip(*tracking_particles))[1])):
                    
                    #                 pass 

                    #             else:

                    #                 if col % 2 == 1:

                    #                     y += height_size / 2

                    #                 try:

                    #                     #particle_to_be_placed = _add_arrays(final_particle_array, white_shadow*illumination_weight)

                    #                     img_array = self.place_particles(img_array, final_particle_array, int(x), int(y))
                                                
                    #                     mask_array = self.place_particles(mask_array, ellipse_array, int(x), int(y))
                                            
                    #                     thickness_scaled = [thickness*scale]

                    #                     x_coords_cluster = x_coords+(y - point[1])
                    #                     y_coords_cluster = y_coords+(x - point[0])
                            
                    #                     tracking_particles.append([int(x), int(y), x_coords_cluster, y_coords_cluster, thickness_scaled]) 
                                    
                    #                 except: pass
                else: 
                    
                    try:
                        
                        #particle_to_be_placed = _add_arrays(final_particle_array, white_shadow*illumination_weight)

                        img_array = self.place_particles(img_array, final_particle_array, int(point[0]), int(point[1]))
            
                        mask_array = self.place_particles(mask_array,ellipse_array,int(point[0]),int(point[1]))
                    
                        tracking_particles.append([point[0], point[1], x_coords, y_coords])
                    
                    except: pass
                            
                    # else:

                    #     try:
                           
                    #         #particle_to_be_placed = _add_arrays(final_particle_array, white_shadow*illumination_weight)

                    #         #plt.imshow(particle_to_be_placed, cmap='gray')
                    #         #plt.axis('off')
                    #         #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/paper2/particle_ellipse4.png", particle_to_be_placed*255)
                    #         #plt.show()
                            
                    #         img_array = self.place_particles(img_array, final_particle_array, int(point[0])+r_x, int(point[1])+r_y)
                                        
                    #         mask_array = self.place_particles(mask_array, ellipse_array, int(point[0])+r_x, int(point[1])+r_y)
                                        
                    #         tracking_particles.append([point[0]+r_x, point[1]+r_y, axes_length[0], axes_length[1], angle])
                    
                    #     except: pass
        
        #img_array = GeneratorUtils().normalise(img_array)
        
        return img_array, mask_array, tracking_particles[1:]

    def place_particles(self, array:np.ndarray, particle:np.ndarray, 
                        x_coordinate:int, y_coordinate:int) -> np.ndarray:
        
        """
        place a particle in the overall image array
        
        Arguments:
            
            "array":  array to place the particle in
            
            "particle array": the particle array
            
            "x-coordinate": where to place particle in x
            
            "y-coordinate": where to place particle in y
            
        Returns:
            
            "array":  final array containing the new particle
            
            """

        # randle_smaller_than_0_x = np.max([0, x_coordinate-int(particle.shape[0]/2)])
        # randle_greater_than_array_size_x = np.min([array.shape[0], x_coordinate-int(particle.shape[0]/2)+particle.shape[0]])
        
        # randle_smaller_than_0_y = np.max([0, x_coordinate-int(particle.shape[0]/2)])
        # randle_greater_than_array_size_y = np.min([array.shape[0], x_coordinate-int(particle.shape[0]/2)+particle.shape[0]])

        # ## using indexing, place particle in the image array
        # array[randle_smaller_than_0_x:randle_greater_than_array_size_x, randle_smaller_than_0_y:randle_greater_than_array_size_y][particle != 0] += particle[particle != 0]
        
        array[x_coordinate-int(particle.shape[0]/2):x_coordinate-int(particle.shape[0]/2)+particle.shape[0], y_coordinate-int(particle.shape[1]/2):y_coordinate-int(particle.shape[1]/2)+particle.shape[1]][particle != 0] += particle[particle != 0]

        return array