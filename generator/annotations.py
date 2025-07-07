import numpy as np
import random
import cv2

class GetAnnotations:

    # Get the id value for a given class_name
    def get_id_for_name(self, data_json, target_name):

        categories = data_json.get("categories", [])

        for category in categories:

            if category.get("name") == target_name:
                
                return int(category.get("id"))
            
        return None  # Return None if the name is not found

    def get_annotations(self, particle_list, particle_type, class_name):
        
        regions = {} # dict for the key 'regions' on data dict
    
        # loop over every particle
        for particle in range(len(particle_list)):
            
            #if particle_type == 'fibrils' or particle_type == 'silica' or particle_type == 'gold' or particle_type == 'polygon'  or particle_type == 'rod':
                                
            shape = dict(name="polygon",
                        all_points_x = list(particle_list[particle][2]),
                        all_points_y = list(particle_list[particle][3])
                            )             

            # Polymers are created using cv2.circles    
            # else:    
            #     cx = int(particle_list[particle][0])
            #     cy = int(particle_list[particle][1])
            #     rx = int(particle_list[particle][2]) 
            #     ry = int(particle_list[particle][3]) 
            #     theta = int(particle_list[particle][4])

            #     angles = np.linspace(0, 2*np.pi, 200)

            #     # Calculate x and y coordinates of points on the ellipse to save as polygon
            #     y_coords = cx + rx * np.cos(angles) * np.cos(theta) - ry * np.sin(angles) * np.sin(theta)
            #     x_coords = cy + rx * np.cos(angles) * np.sin(theta) + ry * np.sin(angles) * np.cos(theta)
                
            #     shape = dict(name="polygon",
            #                     all_points_x = list(x_coords),
            #                     all_points_y = list(y_coords)
            #                     ) 

            
            regions[particle] = dict(
                                    shape_attributes=shape,
                                    region_attributes = {'name': class_name}
                                    )

        return regions

    def get_coco_annotations(self, particle_list, particle_type, count, data_json, class_name):
        
        count_ann = 0
        
        for particle in range(len(particle_list)):
            
            # Polymers are created using cv2.circles
            # if particle_type == "polymer":

            #     cx = int(particle_list[particle][0])
            #     cy = int(particle_list[particle][1])
            #     rx = int(particle_list[particle][2]) 
            #     ry = int(particle_list[particle][3]) 
            #     theta = int(particle_list[particle][4])
                
            #     angles = np.linspace(0, 2*np.pi, 200)
                
            #     y = cx - rx
            #     x = cy - ry
            #     w = 2*rx
            #     h = 2*ry

            #     # Calculate x and y coordinates of points on the ellipse to save as polygon
            #     y_coords = cx + rx * np.cos(angles) * np.cos(theta) - ry * np.sin(angles) * np.sin(theta)
            #     x_coords = cy + rx * np.cos(angles) * np.sin(theta) + ry * np.sin(angles) * np.cos(theta)
                
            #     area = cv2.contourArea(np.array((x_coords, y_coords)).T.astype(np.float32))
                
            #     segmentation = [int(val) for pair in zip(x_coords, y_coords) for val in pair]

            # else:
                
            segmentation = [int(val) for pair in zip(particle_list[particle][2], particle_list[particle][3]) for val in pair]

            area = cv2.contourArea(np.array((particle_list[particle][2], particle_list[particle][3])).T.astype(np.float32))

            x, y, w, h = cv2.boundingRect(np.column_stack((particle_list[particle][2], particle_list[particle][3])).astype(np.int32))

            data_json["annotations"].append({
                "id": int(str(random.randint(0,100)) + str(count) + str(random.randint(0,100)) + str(count_ann) + str(random.randint(0,1000))),
                "category_id": self.get_id_for_name(data_json, class_name),
                "iscrowd": 0,
                "segmentation": [segmentation],
                "bbox": [x, y, w, h],
                "area": area,
                "bbox_mode": 1, 
                "image_id": count
            })   

            count_ann += 1

        return data_json
