import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.spatial.distance import cdist
import pyfastnoisesimd as fns
from typing import Tuple
from numba import jit


@jit
def _subtract(array1, array2):

    subtracted_array = np.subtract(array1 , array2)

    return subtracted_array

@jit
def _normalise(array):
    
    subtracted1 = _subtract(array , np.min(array))

    subtracted2 = _subtract(np.max(array), np.min(array))

    normalised_array = np.divide(subtracted1, subtracted2).astype("float32")
    
    return normalised_array

class GeneratorUtils:

    def perlin_noise(self, row:int, col:int, 
                     frequency:float, octaves:int, 
                     gain:float, lacunarity:float) -> np.ndarray:

        """
        Generate a perlin noise array.
        
        Arguments:
            
            "row":  row length of the array.
            
            "col": column length of the array.
            
            "frequency": frequency of the perlin noise in the array.
            
            "octaves": number of fractal noise layers.
            
            "gain": weighting of each layer compared to previous layer.
            
            "lacunarity": strenght of each layer.
            
            "seed": random seed for the perlin noise.
            
        Returns:
            
            "result":  perlin noise array.
            
        """
        
        shape = [row,col] # shape of the desired array
        
        N_threads = 1 # number of cpu threads
        
        perlin = fns.Noise(numWorkers=N_threads) # create Noise object.
        perlin.frequency = frequency # set perlin noise frequency.
        perlin.noiseType = fns.NoiseType.Perlin # define noise type.
        perlin.fractal.octaves = octaves # set fractal noise layers to 'octaves'
        perlin.fractal.gain = gain # set fractal noise gain to gain.
        perlin.fractal.lacunarity = lacunarity # set lacunarity to lacunarity.
        result = perlin.genAsGrid(shape) # generate result array from params
        
        result = self.normalise(result)
        
        return result

    def simplexfractal_noise(self, row:int, col:int, 
                            frequency:float, octaves:int, 
                            gain:float, lacunarity:float) -> np.ndarray:
        
        """
        Generate a simplex fractal noise array.
        
        Arguments:
            
            "row":  row length of the array.
            
            "col": column length of the array.
            
            "frequency": frequency of the simplex fractal noise in the array.
            
            "octaves": number of fractal noise layers.
            
            "persistence": weighting of each layer compared to previous layer.
            
            "lacunarity": strenght of each layer.
            
            "seed": random seed for the simplex fractal noise.
            
        Returns:
            
            "result":  simplex fractal noise array.
            
        """
        
        shape = [row,col] # shape of the desired array
        
        N_threads = 1 # number of cpu threads
        
        simplexfractal = fns.Noise(numWorkers=N_threads) # create Noise object.
        simplexfractal.frequency = frequency # set noise frequency.
        simplexfractal.noiseType = fns.NoiseType.SimplexFractal # define noise type.
        simplexfractal.fractal.fractalType = 'FBM'
        simplexfractal.fractal.octaves = octaves
        simplexfractal.fractal.gain = gain
        simplexfractal.fractal.lacunarity = lacunarity
        result = simplexfractal.genAsGrid(shape) # generate result array from params
        
        result = self.normalise(result)
        
        return result

    def normalise(self, array:np.ndarray) -> np.ndarray:
        
        """
        
        Normalise array between 0 and 1.
        
        Arguments:
            
            "array":  array to be normalised.
            
        Returns:
            
            "array_norm":  normalised input array.
            
        """
        
        #change the range of an array for all values so they lie between 0 & 1. 
        array_norm = _normalise(array)
        #array_norm = ((array - np.min(array)) / ((np.max(array)) - np.min(array))).astype("float32")
        
        return array_norm

    def vorarr(self, regions:list, vertices:np.ndarray, 
               width:int, height:int, points:np.ndarray, 
               vor, dpi=100) -> np.ndarray:
        
        """
        Convert voronoi diagram from image object to array

        Parameters
        ----------
        regions : voronoi polygons
        
        vertices: coordinatwes of voronoi vertices
        
        width: width of array
        
        height: height of the array
        
        points: voronoi points used to make voronoi diagram
        
        vor: voronoi diagram
        
        dpi: resolution of plotted image

        Returns
        -------
        
        voronoi diagram as a numpy array

        """
        
        fig = plt.Figure(figsize=(width/dpi, height/dpi), dpi=dpi)
        canvas = FigureCanvas(fig)
        
        ax = fig.add_axes([0,0,1,1])
        
        # colorize
        for region in regions:
            polygon = vertices[region]
            ax.plot(*zip(*polygon),'k')

        ax.plot(points[:,0], points[:,1],'.b',markerfacecolor="None",markeredgecolor='None')
        ax.set_xlim(vor.min_bound[0], vor.max_bound[0])
        ax.set_ylim(vor.min_bound[1], vor.max_bound[1])

        canvas.draw()

        get_buffer = np.asarray(canvas.buffer_rgba(), dtype='uint8')[:, :, :3]

        final_vor = get_buffer.reshape(height, width, 3)
        
        return final_vor

    def rgb2gray(self, rgb):

        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

    def voronoi_finite_polygons_2d(self, vor, radius=None):
        """
        Reconstruct infinite voronoi regions in a 2D diagram to finite
        regions.

        Parameters
        ----------
        vor : Voronoi
            Input diagram
        radius : float, optional
            Distance to 'points at infinity'.

        Returns
        -------
        regions : list of tuples
            Indices of vertices in each revised Voronoi regions.
        vertices : list of tuples
            Coordinates for revised Voronoi vertices. Same as coordinates
            of input vertices, with 'points at infinity' appended to the
            end.

        """

        if vor.points.shape[1] != 2:
            raise ValueError("Requires 2D input")

        new_regions = []
        new_vertices = vor.vertices.tolist()

        center = vor.points.mean(axis=0)
        if radius is None:
            radius = vor.points.ptp().max()

        # Construct a map containing all ridges for a given point
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        # Reconstruct infinite regions
        for p1, region in enumerate(vor.point_region):
            vertices = vor.regions[region]

            if all(v >= 0 for v in vertices):
                # finite region
                new_regions.append(vertices)
                continue

            # reconstruct a non-finite region
            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]

            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    # finite ridge: already in the region
                    continue

                # Compute the missing endpoint of an infinite ridge

                t = vor.points[p2] - vor.points[p1] # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal

                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[v2] + direction * radius

                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())

            # sort region counterclockwise
            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]

            # finish
            new_regions.append(new_region.tolist())

        return new_regions, np.asarray(new_vertices)

    def gradient_brightness(self, array:np.ndarray) -> np.ndarray:
        
        """
        create gradient brightness array to change particle or background brightness
        
        Arguments:
            
            "array":  array to change the brightness of
            
        Returns:
            
            "array":  gradient brightness particle array
            
        """
        
        random = np.random.uniform(0,1) # generate random number.
        
        row,col = array.shape[0],array.shape[1] # assign values to row and col variables, which are the shape of the input array.

        dimming = np.random.uniform(0,0.3)   #for the hole image
        
        # row vector equal in length to array dimension, values of array spread evenly between dimming variable and 1.
        # This will be used to give the gradient fall off in brightness in the final array.
        segment = np.arange(dimming,1,(1-dimming)/row,dtype='float32')

        if len(segment) > row:
            
            segment = segment[:row].reshape(row,1)
            
        else:
            
            segment = segment.reshape(row,1)
        
        segment = np.tile(segment,col) # turn segment (500,1) vector into repeating (500,500) verctor, so that we account for all array columns

        if random > 0.5: # if random number greater than 0.5, flip the array so dimming happens in opposite direction
            
            segment = np.flip(segment) # flip the array for blurring in the opposute direction

        array = (array-segment) # increase the brightness by adding array and segment
        
        array = array/np.max(array) # normalise
        
        
        return array

    def get_mask_contours(self, mask:np.ndarray) -> list:

        """
        Get mask contour points using find_countors function from skimage measure. 
        
        Arguments:
            
            "mask":  array with mask for one particle
            
        Returns:
            
            "contours_mask":  list of (x, y) coordinates for particle contour
            
        """
       
        contours_mask = []
  
        contours = find_contours(mask, 0.5)

        for verts in contours:

            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            contours_mask.append(np.array(verts, np.int32))

        return contours_mask

    def find_diameter(self, pts:list) -> Tuple[tuple, tuple, float]:

        """
        use the list of countour coordinates to find the two point with maximum distance, this will be the diameter. 
        
        Arguments:
            
            "pts":  list of countour coordinates for one particle
            
        Returns:
            
            "point1": initial diameter coordinate 

            "point2": end diameter coordinate
            
            "distances": distance between points (diameter)
            
        """
        
        # # Find the index corresponding to the maximum distance
        max_distance = cdist(pts, pts).max()

        # Find the indices of the maximum distance
        indices = np.where(cdist(pts, pts) == max_distance)

        # Extract the indices of the two points with the maximum distance
        point1_index = indices[0][0]
        point2_index = indices[1][0]

        # Extract the actual points from the original array
        point1 = pts[point1_index][0], pts[point1_index][1]
        point2 = pts[point2_index][0], pts[point2_index][1]

        return point1, point2, max_distance