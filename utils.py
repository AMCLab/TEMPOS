import pandas as pd
import numpy as np
from generator.read_dm4 import ReadDm4
from scipy.spatial.distance import cdist
#from skimage.measure import find_contours
from cv2 import findContours
from skimage.measure import label, regionprops_table
from skimage.morphology import convex_hull_image
import ncempy.io.dm as dm
from typing import Tuple
import cv2
import matplotlib.pyplot as plt
import emdfile as emd
import re
import h5py
import pytesseract
from PyQt6 import QtCore, QtGui, QtWidgets
# Set path to tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

class FilePathButton(QtWidgets.QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.full_path = ""
        self.setToolTipDuration(30000)  # Show tooltip for 30 seconds
        self.setMouseTracking(True)

    def set_file_path(self, path):
        self.full_path = path
        self.setToolTip(path)  # Set full path as tooltip
        self.update_display()

    def update_display(self):
        if not self.full_path:
            return
            
        # Calculate available space (subtract 20px for padding/margins)
        available_width = self.width() - 20
        display_text = f"File Loaded: {self.full_path}"
        
        # Elide text if needed
        metrics = QtGui.QFontMetrics(self.font())
        elided_text = metrics.elidedText(
            display_text,
            QtCore.Qt.TextElideMode.ElideLeft,  # Show end of path
            available_width
        )
        
        self.setText(elided_text)

class Utils:

    def load_image_into_numpy_array(self, image:np.ndarray) -> np.ndarray:

        """
        Load a grayscale image into a numpy array.
        
        Arguments:
            
            "image":  grayscale image
            
        Returns:
            
            "training_image": expended image
            
        """

        # The function supports only grayscale images
        assert len(image.shape) == 2, "Not a grayscale input image" 
        last_axis = -1
        dim_to_repeat = 2
        repeats = 3
        grscale_img_3dims = np.expand_dims(image, last_axis)
        training_image = np.repeat(grscale_img_3dims, repeats, dim_to_repeat).astype('uint8')
        assert len(training_image.shape) == 3
        assert training_image.shape[-1] == 3
        return training_image 

    def read_image(self, path:str) -> Tuple[np.ndarray, float]:

        """
        Reads an image with ncempy.io.dm if .dm or with cv2 if .png, .jpg, .tiff.
        Gets resolution (pixel size in nm) if .dm, otherwise sets pixel size to 1.
        
        Arguments:
            
            "path":  path to image
            
        Returns:
            
            "training_image": image loaded and resolution
            
        """

        if (path.endswith(".dm3") or path.endswith(".dm4")):
            # Read .dm3/.dm4 image
            try:
                readDm4 = ReadDm4()
                image, px_to_nm, unit = readDm4.get_image_from_dm4(path)
                
                if unit == 'µm':
                    px_to_nm = px_to_nm*1000
                nm_to_px = 1/px_to_nm

                if image.shape[-1] != 3:  # for gray scale image
                    image= self.load_image_into_numpy_array(image)

            except Exception as err: 
                print("Error in file: ", path) 
                print(err)
                pass
              
        elif path.endswith(".emd"):

            image = emd.read(path)

        else:
            
            image = cv2.imread(path)

            try:

                nm_to_px = self.find_ruler_size(image)
                print('Automatic found 1 nm = ', nm_to_px, 'pixels')

            except Exception as err:  
                print(err)
                print("Couldn't get ruler size, setting 1 pixel = 1 nm")
                nm_to_px = 1.

        return image, nm_to_px

    def get_particle_properties_final_df(self, all_masks:list, all_masks_convex_hull:list) -> pd.DataFrame:

        """
        Gets the lists results from get_particle_properties and concatenates them in a single dataframe.
        Computes aspect ratio, circularity, solidity and convexity and add it to dataframe.
        
        Arguments:
            
            "all_masks":  list of computed regionprops_table values. Each element on the list refers to one particle.
            "all_masks_convex_hull":  list of computed convex hull values. Used to calculate solidity and convexity.
            
        Returns:
            
            "final_df": Pandas Dataframe with all particles in one image
            
        """

        # Concatenate all DataFrames
        all_masks_df = pd.concat(all_masks, ignore_index=True)

        all_masks_convex_hull_df = pd.concat(all_masks_convex_hull, ignore_index=True)
        
        # Get properties of interest
        final_df = all_masks_df[['contours','axis_major_length', 'axis_minor_length', 'eccentricity', 'area']].copy()
        #print('AXIS MAJOR LENGTH',all_masks_df['axis_major_length'], all_masks_df['axis_minor_length'])
        final_df['aspect ratio'] = (all_masks_df['axis_major_length']/all_masks_df['axis_minor_length'])
        final_df['circularity'] = 4*np.pi*all_masks_df['area']/(all_masks_df['perimeter']** 2)
        # solidity and convexity are defined as the ratio of the areas and perimeters, 
        # respectively, between the particle and the smallest convex polygon that encapsulates the particle.
        final_df['solidity'] = (all_masks_df['area']/all_masks_convex_hull_df['area'])
        final_df['convexity'] = (all_masks_df['perimeter']/all_masks_convex_hull_df['perimeter'])
        
        return final_df

    def get_particle_properties(self, mask:np.ndarray, contours:list, all_masks:list, all_masks_convex_hull:list, nm_to_px:float) -> Tuple[list, list]:

        """
        Computes 'area', 'perimeter', 'eccentricity','axis_major_length', 'axis_minor_length' and the convex hull for each detection.
        Appends all values to list.

        Arguments:

            "mask": binary mask for one detection from detectron2
            "all_masks":  list of computed regionprops_table values. Each element on the list refers to one particle.
            "all_masks_convex_hull":  list of computed convex hull values. Used to calculate solidity and convexity.
            "nm_to_px": converts pixel to nm if the image is .dm, otherwise is 1.
            
        Returns:
            
            "all_masks":  appended list of computed regionprops_table values. Each element on the list refers to one particle.
            "all_masks_convex_hull": appended list of computed convex hull values. Used to calculate solidity and convexity.
            
        """
        # Draw mask to get properties
        points = np.array(contours)

        x = points[:, 0]  # Extract x-coordinates
        y = points[:, 1]  # Extract y-coordinates
        draw_polygon = np.column_stack((x, y)).astype(np.int32)   

        cv2.fillPoly(mask, [draw_polygon], color=(255, 255, 255))

        mask = mask[:,:,0]
        
        # Get polygon properties form skimage.measure
        label_img = label(mask)
        
        props = regionprops_table(label_img, properties=('area', 'perimeter', 'eccentricity','axis_major_length', 'axis_minor_length'),)

        # Sometimes if get a very small area by mistake and returns two regions. We need to filter just the first one
        props = {key: np.array([val[0]]) for key, val in props.items()}

        # Finding smallest convex polygon that encapsulates the particle (Convex Hull)
        convex_hull = convex_hull_image(mask)
        
        label_convex_hull = label(convex_hull)
        
        props_convex_hull = regionprops_table(label_convex_hull, properties=('area','perimeter'),)
        # Sometimes if get a very small area by mistake and returns two regions. We need to filter just the first one
        props_convex_hull = {key: np.array([val[0]]) for key, val in props_convex_hull.items()}
        
        # Convert to DataFrame and add a column for the detection index (optional)
        df = pd.DataFrame(props)

        # Add list of contours as column on dataframe
        df['contours'] = [[] for _ in range(1)]
        df['contours'] = df['contours'].astype(object)
        df.at[0, 'contours'] = contours

        df['area'] = df['area']/(nm_to_px)**2
        df['perimeter'] = df['perimeter']/nm_to_px
        df['axis_major_length'] = df['axis_major_length']/nm_to_px
        df['axis_minor_length'] = df['axis_minor_length']/nm_to_px
        
        df_convex_hull = pd.DataFrame(props_convex_hull)
        
        # Append to the list
        all_masks.append(df)
        all_masks_convex_hull.append(df_convex_hull)

        return all_masks, all_masks_convex_hull
    
    def find_diameter(self, pts:list) -> Tuple[tuple, tuple, np.float64]:

        """
        Finds the two points with greatest distance from each other, this will be the major axis (diameter) of a detected particle.

        Arguments:

            "pts": all contour points of the particle.
            
        Returns:
            
            "point1": First point with maximum distance from point2.
            "point2": Second point with maximum distance from point1.
            "distance": Discance between the two points.

        """
        
        # # Find the index corresponding to the maximum distance
        distance = cdist(pts, pts).max()

        # Find the indices of the maximum distance
        indices = np.where(cdist(pts, pts) == distance)

        # Extract the indices of the two points with the maximum distance
        point1_index = indices[0][0]
        point2_index = indices[1][0]

        # Extract the actual points from the original array
        point1 = pts[point1_index][0], pts[point1_index][1]
        point2 = pts[point2_index][0], pts[point2_index][1]

        return point1, point2, distance

    def get_mask_contours(self, mask:np.ndarray) -> list:

        """
        Finds the contour points of a binary mask.

        Arguments:

            "mask": Binary mask from Detectron2.
            
        Returns:
            
            "contours_mask": List with the contour points.

        """
            
        #contours_mask = []
        mask = np.ascontiguousarray(mask)
        contours = findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        # for verts in contours:
        #     # Subtract the padding and flip (y, x) to (x, y)
        #     verts = np.fliplr(verts) - 1
        #     contours_mask.append(np.array(verts, np.int32))

        hierarchy = contours[-1]
        if hierarchy is None:  # empty mask
            return [], False
        has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
        contours = contours[-2]
        contours = [x.flatten() for x in contours]
        # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
        # We add 0.5 to turn them into real-value coordinate space. A better solution
        # would be to first +0.5 and then dilate the returned polygon by 0.5.
        contours = [x + 0.5 for x in contours if len(x) >= 6]
        
        return contours, has_holes


    def get_dm4_info(self, path:str) -> list:

        """
        Get all information from .dm file metadata, except "data", which is the image and "filename".

        Arguments:

            "path": Path of .dm file.
            
        Returns:
            
            "myData": List with the .dm metadata information.

        """

        myData = list()

        dm4_file = dm.dmReader(path) #read the dm3 file

        dm4_file.pop('data')
        dm4_file.pop('filename')

        dm4_file['coords'] = dm4_file['coords'][0]
        dm4_file['pixelSize'] = str(dm4_file['pixelSize'][0])
        dm4_file['pixelUnit'] = str(dm4_file['pixelUnit'][0])

        myData.append(dm4_file)

        return myData
    
    def convert_metric_to_nm(self, input_str):
        # Extract numeric value and unit using regex
        match = re.match(r"^\s*([\d.]+)\s*([a-zA-Zμ]+)\s*$", input_str)
        if not match:
            raise ValueError(f"Invalid format: {input_str}")
        
        number_str, unit = match.groups()
        number = float(number_str)
        
        # Define conversion factors to nanometers
        unit_conversion = {
            'nm': 1,
            'nam': 1, # it reads wrong sometimes
            'am': 1, # it reads wrong sometimes
            'nanometer': 1,
            'μm': 1000,
            'µm': 1000,  # Alternative spelling
            'um': 1000,  # Alternative spelling
            'micron': 1000,
            'micrometer': 1000,
            'µ': 1000,  # Handle single-character mu symbol
            'µ': 1000  # Handle single-character mu symbol
        }
        
        # Normalize unit to lowercase and check validity
        unit = unit.lower().replace('μ', 'u')  # Normalize micro symbol
        # if unit not in unit_conversion:
        #     raise ValueError(f"Unsupported unit: {unit}")
        
        # Convert to nanometers and return as integer
        return int(number * unit_conversion[unit])

    def find_ruler_size(self, image):
        # The bar can be either black or white
        if (image == 255).any():
            rule_bar_color = 255
            img_color = 0
        if (image == 0).any():
            rule_bar_color = 0
            img_color = 255

        print(rule_bar_color)
        print(img_color)

        ruler_only_img = image.copy()
        # Remove all non white/black values
        ruler_only_img[ruler_only_img != rule_bar_color] = img_color

        #Invert image if the bar is black
        if rule_bar_color == 0:

            ruler_only_img = 255 - ruler_only_img

        # Find non zero indic-es
        nonzero_indx = np.argwhere(ruler_only_img) 
        # Sort indx
        nonzero_indx = np.sort(nonzero_indx, axis= 0)
        # Get first and last pixel position -> thats the size of our bar
        start, end = (nonzero_indx[0], nonzero_indx[-1])
        # Calculate size in y axis
        rule_size = end[1] - start[1]

        # read characteres and get rule value
        d = pytesseract.image_to_string(ruler_only_img, config="--psm 10")
        d.strip()
        d = self.convert_metric_to_nm(d)
        nm_to_px = rule_size/d

        return nm_to_px
# def random_colors(N, bright=True):
    #     """
    #     Generate random colors.
    #     To get visually distinct colors, generate them in HSV space then
    #     convert to RGB.
    #     """
    #     brightness = 255 if bright else 180
    #     hsv = [(i / N + 1, 1, brightness) for i in range(N + 1)]
    #     colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    #     random.shuffle(colors)

        # return colors