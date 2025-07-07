from read_dm4 import ReadDm4
from generator_utils import GeneratorUtils
import cv2
import os
import matplotlib.pyplot as plt

readDm4 = ReadDm4()

bg_image_path = 'C:\\Users\\Natalia\\OneDrive\\Documents\\TEMData\\'

for subdir, dirs, files in os.walk(bg_image_path):

    for item in files:
        if item.endswith('.dm3') or item.endswith('.dm4'):
            print(item)
            fullpath = os.path.join(subdir, item)

            img, resolution, unit = readDm4.get_image_from_dm4(fullpath)

            img = cv2.resize(img,
                            dsize=(1024, 1024),
                            interpolation=cv2.INTER_NEAREST)

            img = GeneratorUtils().normalise(img)

            print(fullpath)

            #cv2.imwrite(bg_image_path + 'png_files\\' + item +'.png', 255*img)
            plt.imshow(img, cmap='gray')
            plt.show()