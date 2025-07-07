import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from .generator_utils import GeneratorUtils
import cv2
import random
from numba import jit
import matplotlib.pyplot as plt

@jit
def _find_index(array:np.ndarray, condition:int) -> np.ndarray:
        
    ind = np.where(array>condition) # index of values above inital_img_size+1000, 1st dim

    return ind

@jit
def _create_zeros_array(size:tuple, dtype:np.dtype):

    zeros_array = np.zeros(size, dtype)

    return zeros_array

@jit
def _add_arrays(array1, array2):

    added_array = np.add(array1, array2)

    return added_array

class Grid: 
    
    def remove_values(self, array:np.ndarray, inital_img_size:int) -> np.ndarray:

        ind = _find_index(array[:,0], inital_img_size+1000)

        array = np.delete(array[:], ind,axis=0) # delete these values from array
        
        ind = _find_index(array[:,1], inital_img_size+1000) # index of values above inital_img_size+1000, 2nd dim
        
        array = np.delete(array[:], ind,axis=0) # delete these values from array
        
        ind = _find_index(array[:,0], 500) # index of values below 500, 1st dim
        
        array = np.delete(array[:], ind,axis=0) # delete these values from array
        
        ind = _find_index(array[:,1], 500) # index of values below 500, 2nd dim
        
        array = np.delete(array[:], ind,axis=0) # delete these values from array

        return array
        

    def define_shadow_lacey_carbon(self, array:np.ndarray) -> np.ndarray:
        
        """
        
        Define 'shadow' region of the lacey-carbon using the sobel filter.
        
        Arguments:
            
            "array":  lacey-carbon array to find shadows from
            
            "rotate": angle to rotate the shadows at.
            
        Returns:
            
            "blur": shadow array for inputted lacey-carbon array.
            
            """
            
        gaussian_kernel = 21
        gaussian_deviation = 2

        # vertical sobel filter array
        vertical_filter = cv2.Sobel(array, cv2.CV_32F, 1, 0)

        # horizontal sobbel filter array
        horizontal_filter = cv2.Sobel(array, cv2.CV_32F, 0, 1)

        # combine horizontal and vertical to orientation array
        orientation = np.arctan2(horizontal_filter,vertical_filter)*(180/np.pi) # Element-wise arc tangent of x1/x2 

        final_array = GeneratorUtils().normalise(orientation) # normalise array

        blur = cv2.GaussianBlur(final_array, (gaussian_kernel,gaussian_kernel), gaussian_deviation) # blur the image

        blur = GeneratorUtils().normalise(blur) # normalise image

        return blur


    def lacey_carbon_grid(self, inital_img_size:int) -> np.ndarray:
        
        """
        Generate the lacey carbon grid.
        
        Arguments:
            
            None.
            
        Returns:

            "binary_image": binary lacey-carbon grid template. Can be use later to individually change color of holes/grid
            
            "final_template": the final lacey-carbon grid template.
                
            "points": the voronoi points used to create the grid. 

        """
        
        if np.random.uniform(0,1) < 0.5:

            kernel_size = random.randrange(29, 71, 2) # thickness of voronois, the bigger the number, the thicker the lines (smaller carbon holes)
        
            gaussian_blur = random.randrange(61, 101, 2) # blur to round the lines, the bigger the blur, the thicker the line (smaller carbon holes)
    
        else:
        
            kernel_size = 29

            gaussian_blur = 71

        points = np.random.rand(400, 2) # generate series of random points
        
        vor = Voronoi(points) # compute Voronoi tesselation
     
        # voronoi_finite_polygons_2d function from https://stackoverflow.com/a/20678647/425458
        regions, vertices = GeneratorUtils().voronoi_finite_polygons_2d(vor)
        
        # convert plotting data to numpy array
        arr = GeneratorUtils().vorarr(regions, vertices, width=inital_img_size +1000, height=inital_img_size +1000,points=points,vor=vor)
        
        arr = arr[500:inital_img_size + 500, 500:inital_img_size + 500] # cut 2048 x 2048 segment from array

        arr = GeneratorUtils().normalise(arr) # normalise array
    
        arr = np.mean(arr, -1) # convert array from rgb to grayscale
        
        arr = np.where(arr>0.5,1,0) # binarise array using thrreshold of 0.5
    
        arr = (1 - arr).astype('uint8') # invert array values

        #plt.imshow(arr, cmap='gray')
        #plt.axis('off')
        #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/Thesis/figures/bg_lacey0.png", arr*255)
        #plt.show()

        ## perform dilations and guassian blurring on the voronoi grid

        kernel_centre = int(kernel_size / 2) # centre of kernel based on the kernel_size variable

        kernel = _create_zeros_array((kernel_size, kernel_size), np.uint8) # create kernel based on the kernel_size variable

        kernel = cv2.circle(kernel, (kernel_centre, kernel_centre), int(kernel_size/2) - 1, 1, -1) # create circular kernel

        dilation = cv2.dilate(arr,kernel,iterations = 1) # apply dilation using the circular kernel

        binary_img = cv2.GaussianBlur(dilation, (gaussian_blur,gaussian_blur), 21) # blur the dilated array
        
        #plt.imshow(binary_img, cmap='gray')
        #plt.axis('off')
        #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/Thesis/figures/bg_lacey1.png", binary_img*255)
        #plt.show()

        final_template = self.define_shadow_lacey_carbon(binary_img) # create the shadow array for lacey carbon
        
        #plt.imshow(final_template, cmap='gray')
        #plt.axis('off')
        #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/Thesis/figures/bg_lacey2.png", final_template*255)
        #plt.show()

        points[:,0] = np.round(points[:,0]*inital_img_size+1000) # re-range voronoi points
        
        points[:,1] = np.round(np.abs(points[:,1]-1)*inital_img_size+1000) # re-range voronoi points
        
        points = self.remove_values(points, inital_img_size) - 500 # remove vals outside final image range
        
        return binary_img, final_template, points

    def holey_carbon_grid(self, inital_img_size:int) -> np.ndarray:
        """
        
        Create holey carbon grid
        
        Arguments:
            
            "inital_img_size":  array size to be generates.
            
        Returns:
            
            "ternary_img": voronoi array colored with 0 (carbon holes), 1 (holes contours) or 2 (background). Can be use to change the color of the background individually

            "final_template": same voronoi array with color and shadows.

            "points": valid voronoi points.
            
            """

        kernel_size = 100

        gaussian_blur = 81    #try instead of new grid

        points = np.random.rand(400, 2) # generate series of random points
        
        vor = Voronoi(points) # compute Voronoi tesselation

        # voronoi_finite_polygons_2d function from https://stackoverflow.com/a/20678647/425458
        regions, vertices = GeneratorUtils().voronoi_finite_polygons_2d(vor)
        
        # convert plotting data to numpy array
        arr = GeneratorUtils().vorarr(regions, vertices, width=inital_img_size+1000, height=inital_img_size+1000,points=points,vor=vor)

        plot_vorr_array = False
        if plot_vorr_array == True:
            # Create a copy of the Voronoi array to add points
            arr_with_points = arr.copy()

            # Scale points to original array dimensions and adjust for OpenCV's coordinate system
            width = inital_img_size + 1000
            height = inital_img_size + 1000

            # Scale points and invert Y-axis
            scaled_x = (points[:, 0] * width).astype(int)
            scaled_y = (height - 1 - (points[:, 1] * height)).astype(int)  # Flip Y-axis

            # Draw red circles (3px radius) on valid points
            for x, y in zip(scaled_x, scaled_y):
                cv2.circle(
                    arr_with_points, 
                    (x, y), 
                    radius=5, 
                    color=(0, 0, 255),  # Red in BGR
                    thickness=-1
                )

            # Create a copy of the Voronoi array to add points
            arr_with_points = arr_with_points[500:inital_img_size+500,500:inital_img_size+500]

            #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/Thesis/figures/bg_holey_voronoi_with_points.png", GeneratorUtils().normalise(arr_with_points)*255)

        arr = arr[500:inital_img_size+500,500:inital_img_size+500] # cut 2048 x 2048 segment from array

        #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/Thesis/figures/bg_holey_voronoi.png", GeneratorUtils().normalise(arr)*255)

        arr = GeneratorUtils().normalise(arr) # normalise array between 0 and 1
    
        arr = np.mean(arr, -1) # convert array from rgb to grayscale
        
        arr = np.where(arr>0.5,1,0) # binarise array using thrreshold of 0.5
   
        arr = (1 - arr).astype('uint8') # invert array values

        ## perform dilations and guassian blurring on the voronoi grid

        kernel_centre = int(kernel_size / 2) # centre of kernel based on the kernel_size variable

        kernel = _create_zeros_array((kernel_size, kernel_size), np.uint8) # create kernel based on the kernel_size variable

        kernel = cv2.circle(kernel, (kernel_centre, kernel_centre), int(kernel_size/2) - 1, 1, -1) # create circular kernel

        dilation = cv2.dilate(arr,kernel,iterations = 1) # apply dilation using the circular kernel

        #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/Thesis/figures/bg_holey_dilation.png", GeneratorUtils().normalise(dilation)*255)

        blur = cv2.GaussianBlur(dilation, (gaussian_blur,gaussian_blur), 21) # blur the dilated array

        #plt.imshow(blur, cmap='gray')
        #plt.axis('off')
        #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/Thesis/figures/bg_holey_blur.png", GeneratorUtils().normalise(blur)*255)
        #plt.show()

        ######################
        ######################
        # dark gray countour

        # vertical sobel filter array
        vertical_filter = cv2.Sobel(blur, cv2.CV_32F, 1, 0)

        # horizontal sobbel filter array
        horizontal_filter = cv2.Sobel(blur, cv2.CV_32F, 0, 1)

        #plt.imshow(vertical_filter, cmap='gray')
        #plt.axis('off')
        #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/Thesis/figures/bg_holey_vertical_filter.png", GeneratorUtils().normalise(vertical_filter)*255)
        #plt.show()

        #plt.imshow(horizontal_filter, cmap='gray')
        #plt.axis('off')
        #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/Thesis/figures/bg_holey_horizontal_filter.png", GeneratorUtils().normalise(horizontal_filter)*255)
        #plt.show()

        mask = _create_zeros_array((blur.shape[0],blur.shape[1]), dtype=np.float32) # empty mask array

        mask[vertical_filter != 0] = 1 # set mask = 1 where vertical filter is not equal to 0
    
        mask[horizontal_filter != 0] = 1 # set mask = 1 where horiz filter is not equal to 0

        #plt.imshow(mask, cmap='gray')
        #plt.axis('off')
        #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/Thesis/figures/bg_holey_mask.png", GeneratorUtils().normalise(mask)*255)
        #plt.show()

        contour_kernel_size = 23

        contour_kernel_centre = int(contour_kernel_size / 2) # centre of kernel based on the contour_kernel_size variable

        contour_kernel = _create_zeros_array((contour_kernel_size, contour_kernel_size), np.uint8) # create kernel based on the contour_kernel_size variable

        contour_kernel = cv2.circle(contour_kernel, (contour_kernel_centre, contour_kernel_centre), int(contour_kernel_size/2) - 1, 1, -1) # create circular kernel

        #dilation = cv2.dilate(arr,contour_kernel,iterations = 1) # apply dilation using the circular contour_kernel

        mask = cv2.dilate(mask,contour_kernel,iterations = 1) # apply dilation using the circular contour_kernel

        mask[blur == 0] = 0 # remove gray from inside blob

        #plt.imshow(mask, cmap='gray')
        #plt.axis('off')
        #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/Thesis/figures/bg_holey_mask_dilated.png", GeneratorUtils().normalise(mask)*255)
        #plt.show()

        final_template = self.define_shadow_lacey_carbon(blur) # create the shadow array for lacey carbon

        #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/Thesis/figures/bg_holey_shadow.png", GeneratorUtils().normalise(final_template)*255)

        ternary_img = _add_arrays(blur, mask)

        # plt.imshow(ternary_img, cmap='gray')
        # plt.show()
        #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/Thesis/figures/bg_holey_ternary_img.png", GeneratorUtils().normalise(ternary_img)*255)

        final_template = cv2.subtract(final_template, (mask*0.2))

        # points[:,0] = np.round(points[:,0]*inital_img_size+1000) # re-range voronoi points
        
        # points[:,1] = np.round(np.abs(points[:,1]-1)*inital_img_size+1000) # re-range voronoi points
        
        # points = self.remove_values(points, inital_img_size) - 500 # remove vals outside final image range
        
        #plt.imshow(final_template, cmap='gray')
        #plt.axis('off')
        #cv2.imwrite("C:/Users/Natalia/OneDrive/Documents/Thesis/figures/holey_carbon_ilumination_and_contour.png", GeneratorUtils().normalise(final_template)*255)
        #plt.show()

        return ternary_img, final_template
