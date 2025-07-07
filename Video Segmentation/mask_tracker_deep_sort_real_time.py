import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
# import tensorflow as tf
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
from typing import Tuple
from scipy.spatial.distance import cdist
from skimage import measure
import json
import math

""" image operations import """

import cv2
import numpy as np
import matplotlib.pyplot as plt

""" import some common detectron2 utilities """
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.colormap import random_color
import matplotlib.colors as mplc

from deep_sort_realtime.deepsort_tracker import DeepSort


""" command line flags definition """
flags.DEFINE_string('f', '', '')
flags.DEFINE_string('video', '', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', '', 'path to output video')
flags.DEFINE_string('yaml', '', 'path to yaml file with model configuration.')
flags.DEFINE_string('device', '', 'device to run the model on. (cpu or cuda)')
flags.DEFINE_string('model_weights', '', 'path pre trained model weights.')
flags.DEFINE_string('log_details', 'false', 'should log details to console or not. (true or false)')
flags.DEFINE_string('info_panel', 'true', 'should log details to information panel at top left corner or not. (true or false).\
                                 Slows down drawing process to some extent')
flags.DEFINE_string('json_path', '', 'path to save json with particle info by frame')




def find_diameter(pts:list) -> Tuple[tuple, tuple, np.float64]:

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


def main(_argv):

    
    cfg = get_cfg()
    cfg.merge_from_file(FLAGS.yaml)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.DEVICE = FLAGS.device
    cfg.MODEL.WEIGHTS = FLAGS.model_weights
    cfg.SOLVER.IMS_PER_BATCH = 1  # This is the real "batch size" commonly known to deep learning people
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size".
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    tic = time.time()
    print('Start time: ', tic)
    # Set up Deep SORT tracker
    tracker = DeepSort(max_age=60, n_init=5, nn_budget = 200 , nms_max_overlap=0.5, max_iou_distance=0.5, bgr=False, max_cosine_distance=0.8)
    
    video_path = FLAGS.video
   

    """ begin video capture """
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)


    """ position of top left corner dsplay block """
    x, y, w, h = 10, 10, int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)/4.5), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)/2)
    

    length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Total frames in video: ', length)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    print('fps ', fps)
    codec = cv2.VideoWriter_fourcc(*"mp4v")

    """ 
    Video output configuration:
    output path,
    codec, 
    fps of output video,
    width and height of output frames ( must match resultant frame )
    """
    out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height), isColor=True)

    """
    list of all classes in the maskRCNN model in the same order as used for training the model.
    """
    class_names =  ["particle"]

    """
    list all classes that can be tracked by the tracker.
    If you want to skip certain classes from the model for tracking then
    exclude them from below list, else both class_names and allowed_classes
    will be same.
    """
    allowed_classes =  ["particle"]

    frame_num = 0
    draw_particle_path = []

    json_path = FLAGS.json_path
    save_particle_info = dict()

    """ while video is running """
    while True:
        """
        Read frames from video
        """
        return_value, frame = vid.read()
        if not return_value:
            print('Video Ended!!')
            break
        frame_num +=1
        # print('frame_' + str(frame_num))
        start_time = time.time()

        """
        Perform prediction on frame and extract the following details:
            1. bounding boxes
            2. scores
            3. classes
            4. masks

        
        These are the infromation provided by all Instance Segmentation models.
        So, this code can be used to track on all Instance segmentation models.
        And, if an Instance Segmented video can be tracked then, Semantic Segmented
        videos can be tracked too with some tweaks in code.

        For maskRCNN models built with other frameworks can also be tracked with same code
        but for drawing on images detectron2 default functions are used so detectron2 will also
        be needed along with the other framework.
        """
        outputs = predictor(frame) # maskRCNN prediction in frame
        #instances = outputs["instances"].to("cpu")

        bboxes = outputs['instances'].get_fields()['pred_boxes'].tensor.cpu().numpy() // 1
        scores = outputs['instances'].get_fields()['scores'].cpu().numpy()
        classes = outputs['instances'].get_fields()['pred_classes'].cpu().numpy()
        masks = outputs['instances'].get_fields()['pred_masks'].cpu().numpy()

        particle_info = []

        if len(bboxes) > 0:
            bboxes = np.column_stack((bboxes[:, 0], bboxes[:, 1], bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]))

            # Deep SORT expects list of [ [x1,y1,x2,y2], score, class ]
            confidence_threshold = 0.7

            detections = [
                (box, score, cls)
                for box, score, cls in zip(bboxes, scores, classes)
                if score >= confidence_threshold
            ]


            # Update tracker
            tracks = tracker.update_tracks(detections, frame=frame)

            
            
            # Draw results
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()  # left, top, right, bottom
                #cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0,255,0), 2)
                cv2.putText(frame, f'ID {track_id}', (int(ltrb[0]), int(ltrb[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                
                # Find closest detection box (simple L2 distance)
                track_box = track.to_ltrb()
                distances = [np.linalg.norm(np.array(track_box[:2]) - np.array(box[:2])) for box in bboxes]
                match_idx = np.argmin(distances)
                mask = masks[match_idx]

                mask = (mask.astype(np.uint8)) * 255  # values now 0 or 255

                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                props = measure.regionprops(mask.astype(np.uint8))[0]
                
                major_axis_lenght = props.major_axis_length
                #print('Major axis length: ', major_axis_lenght)

                # y0, x0 = props.centroid
                # orientation = props.orientation
                # x2 = x0 - math.sin(orientation) * 0.5 * props.axis_major_length
                # y2 = y0 - math.cos(orientation) * 0.5 * props.axis_major_length
                # #Draw a line with the major axis length
                # #Expand mask to 3D 
                # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                # mask = cv2.line(mask, (int(x0), int(y0)), (int(x2), int(y2)), (255, 0, 0), 1)

                # #save mask
                # cv2.imwrite(os.path.join("DeepSortMask/masks", f"mask_{track_id}.png"), mask)

                #_, _, diameter = find_diameter(np.array(contours))

                point_x = int((ltrb[2] - ltrb[0])/2 + ltrb[0])
                point_y = int((ltrb[3] - ltrb[1])/2 + ltrb[1]) 
                draw_particle_path.append((point_x, point_y))
                
                # Draw contours on the frame
                cv2.drawContours(frame, contours, -1, (0, 0, 255), thickness=5)  # Green outline

                for center in draw_particle_path:               
                    
                    cv2.circle(frame, center, radius=2, color=(0, 0, 255), thickness=-1)  
                    
                particle_info_dict = {'particle':str(track.track_id),
                                      'particle_centre':(point_x, point_y),
                                      'particle_diameter':major_axis_lenght}
                
                particle_info.append(particle_info_dict)
        
        save_particle_info[frame_num] = particle_info
        out.write(frame)         # save video file 

        del frame
        
        with open(json_path, "w") as f:

            f.write(json.dumps(save_particle_info, indent=2)) 

    #cv2.destroyAllWindows()
    toc= time.time()
    print('End time: ', toc)
    print("Total time: ", (toc - tic))

if __name__ == '__main__':
    # app.run(main)
    try:
        app.run(main)
    except Exception as e:
        print("ERROR!, Something went wrong while trying to run the code.\nPlease refer to the following error msg" + repr(e))

