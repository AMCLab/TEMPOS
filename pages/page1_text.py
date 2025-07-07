about = """
This App was developed with the intention of facilitate the use of Machine Learning models for TEM applications. \
It allows the creation of synthetic data, application of CTF to images, training and evaluation of deep learning models \
for nanoparticle segmentation, automated segmentation of images, including in the .dm3 and .dm4 format, with particle measurements. \
An additional page was created to export your files (with or without predictions) in the [Datasette](https://datasette.io/) format. \
Here you can find a detailed explanation of each of the pages:


## Create Data

Generate synthetic data that mimics your own TEM experimental data. After selecting all the parameters for the data, click in "Generate Image" to visualise an\
example of image. Please note there are several random parameters for generate the image, so even with the same setting your images will differ. \
When satisfied with the visual of your final image, click on "Add Setting" and the app will save as many settings as you want. They can be retrieved in the "My settings" option in the\
 "Generate Data" page, where you can find options to create your dataset. Possible Settings:

- Grid type: 

    1. Lacey carbon; 
    2. Holey carbon; 
    3. Continuous carbon; 
    4. Import a directory with experimental background-only images (some accepted formats are .jpg, .tiff, .dm3, .dm4).

- Particle type: 

    1. Polymer: Ellipses with small variation between minor and major axis. One point can generate a cluster with 50% probability;
    2. Silica: Rounded but irregular shape, median variation between minor and major axis;
    3. Shapes: Random shapes with 3 to 7 vertices;
    4. Gold: Similar to silica, but one particle can generate a random size cluster of particles with 20% probability;
    5. Rod: "Capsule shaped" particles,  can generate a random size cluster of particles if the set number of particles is smaller than 3;
    6. Fibrils: Thin rectangles with low contrast.  

- Initial i: Represents by how much the image will be cropped, high values of Initial i leads to a more "zoomed in" image (lower magnification). \
    A random number will be chosen from the interval for each image generated.

- Particle size range: Range of sizes that the particle can have.

- Number of potential particles: Approximate number of particles on the image. It is not exact once clusters can be generated in some types of particles\
    and there is a non-overlapping rule that can exclude some potential points.

- Particle Weight: Background and Particle arrays will be added to form the final image. Particle weight is the \
    weight of the particle array in the sum.

- Particle Noise: Noise to be applied on top of the particles. None or Simplex Fracta from \
    [PyfastNoisesimd](https://pypi.org/project/pyfastnoisesimd/). \
    Noise and Particle arrays will be added to form the final particle array. Noise weight is the weight of the noise in the sum.

- Image Noise: Noise to be applied on top of final image. None, Perlin or Simplex Fracta from [PyfastNoisesimd](https://pypi.org/project/pyfastnoisesimd/). \
    Noise array will be added to the final image. Image noise weight is the weight of the noise in the sum.

- Reescale Intensity: If checked, the intensity of the final image will be reescaled using \
    [Skimage Reescale Intensity](https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.rescale_intensity) \
    with explicit min/max in_range of [1, 99], given a probability.

- Add Class Name: For training a model with more than one class, when interested in classify between two or more types of particle. \
    Choose an unique name for each setting added.


## Generate Data

In the Generate Data page, you can generate your synthetic dataset. Choose from "My settings" added in the Create Data page, \
"Import a Json" following the default format or choose the "Presets" option to quickly generate some synthetic data.

"Export My Settings as Json": save your settings as a .json if you wish to later reproduce or share your settings.

"Remove Setting": if you have added a setting that is not to your like, you can simply remove it. When the option is checked, a \
drag-down menu will appear with the options you can remove. Click "Remove" to remove setting.

Choose the number of images to be generated for each one of your settings and the train/validation ratio. \
Images are saved with size 2048x2048.

Select the directory you want to save your dataset on. 

Click Generate to start.

Outputs:

1. A new folder will be created with a name that follows the pattern: YearMonthDayHourMinuteSecond
2. Subdirectory: Train, with the train images and the annotations instances_train.json \
3. Subdirectory: Val, with the validation images and annotations instances_val.json.


The images and annontations follow the COCO dataset format and can be used to train any machine learning model that \
uses COCO for instance segmentation. Another .json is generated (via_region_data.json) for each dataset, that follows \
the [VIA](https://www.robots.ox.ac.uk/~vgg/software/via/) (VGG Image Annonator) structure and can be used to train the Mask RCNN model.

## Apply CTF

Apply a CTF to a specific image or to all images in a directory. The function is taken from \
Asarnow, D., Palovcak, E., Cheng, Y. UCSF pyem v0.5. Zenodo https://doi.org/10.5281/zenodo.3576630 (2019). \
An example of use can be found [here](https://github.com/geoffwoollard/learn_cryoem_math/blob/master/nb/2d_class_fourier.ipynb). \
The following parameters can be controlled:
    
1. Pixel size: Size of each pixel in (Å).
2. 1st principal underfocus distance (Defocus) (Å). If a range is selected, the algorithm will \
apply evenly spaced numbers over the total number of images in the direcotry using [numpy linspace](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html). \
so that the first image will have the minimum value and the last image will have the maximum value.
3. Angle of astigmatism (deg): from x-axis to azimuth. If a range is selected, the algorithm will \
apply evenly spaced numbers over the total number of images in the direcotry using [numpy linspace](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html). \
so that the first image will have the minimum value and the last image will have the maximum value.
4. Phase shift (deg).
5. Microscope acceleration potential (kV).
6. Amplitude contrast in [0, 1.0].
7. Spherical aberration (mm).
8. B-factor, divided by 4 in exponential, lowpass positive.
9. Hard low-pass filter (Å), should usually be Nyquist.

In the Add CTF to Original, you can combine the CTF results with your synthetic data with a given weight, \
creating a more realistic image. You can save the image displayed on the app \
pressing the disk icon, or choose a direcoty to apply the CTF. For the second option, you can overwrite the original files or save as copies.

## Train Model

For now, the app supports the [Detectron2](https://github.com/facebookresearch/detectron2) model for instance segmentation. 

Choose the dataset path to use in training, the images and the annotation .json should be in the same directory \
and the annotation HAS to be named instances_train.json and follow the [COCO format](https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html). \
After selecting a path, your dataset will be locally [registered](https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html#register-a-dataset) \
as a COCO dataset.

Next, you should inform the number of classes of your dataset and on which device you want the model to be trained. Cuda will be faster and is recommended if your machine has support for it. Make sure there \
is enough memory available once the images are size 2048x2048.

Test dataset is optional, should follow COCO format and the annotation json should be named instances_val.json.

On advanced settings you can control:

1. Batch size (how many images are used at each iteration). Can be increased if the gpu has more than 16gb.
2. Learning Rate
3. Model initial Weights: the model we use is pre-trained with COCO dataset and fine-tunned with our synthetic data. \
You can change which pretrained model to use. More available weights [here](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md) \
or use one of your own models.
4. Number of Maximum iteration: similar to number of epochs. Can stop sooner if the model stops improving.
5. RoI Head Batch Size: Sample a subset of proposals coming out of RPN to calculate cls and reg loss during training

Start to start your training. You can view an example of training data on the left hand side of the window and the logs \
and training progress on the right hand size of the window.

Outputs:

1. Creates an "output" directory with checkpoints (.pth files) and a model_final.pth


## Evaluate Model

This section will evaluate a model on a given dataset, it will return the AP (Average Precision) and mAP (mean Average Precision) \
and therefore it needs both images and their ground-truth annotations in the same directory. \
The annotation json has to be named instances_val.json and be in a COCO format for instance segmentation.

Select a model to be loaded and the right number of classes with which the model was trained.

The IoU threshold is the minimum confidence to be accepted as a detection, following the \
[IoU](https://www.v7labs.com/blog/intersection-over-union-guide) metrics. \
High Threshold value will result in less false detections, but the model might miss more particles.

Select the path where your validation dataset is, it is mandatory to have a instance_val.json with the ground truth annontations. \
Can be the synthetic validation data generated in the Generate Data page, or a experimental dataset with hand annotations if you wish to \
evaluate the model on real data.

Press Evaluate to start the evaluation process.

## Segment Images

Load model for prediction and the right number of classes the model was trained on.

The IoU threshold is the minimum confidence to be accepted as a detection, following the \
[IoU](https://www.v7labs.com/blog/intersection-over-union-guide) metrics. \
High Threshold value will result in less false detections, but the model might miss more particles.

You can load an individual image to visualise the predictions results.

Select .json used for training (optional): load the instances_train.json used for training so that the \
classes names can me mapped. Otherwise they will be displayed as numbers. 

Press Plot Segmentation to visualise results, and personalise the display showing/hidding plot axis, predictions \
and diameters. Shown on the left hand side of window.

On the right hand size, you can visualise the statistics results as histograms for the particles detected on the image:

1. Major Axis Length
2. Minor Axis Length
3. Eccentricity
4. Area
5. Aspect Ratio
6. Circularity
7. Solidity
8. Convexity

Note that if your file is .dm3 or .dm4 the pixel size information will be collected from the metadata and the \
results will be shown in nm. Otherwise the values will be in pixels.

You can save the results pressing the disc icon on the bottom, optionally with transparent background and the set DPI value.

"Apply Segmentation for All Images in Directory" to chose the directory path to segment. It will use the model, number of classes \
and IoU Threshold set. Creates a directory named "segmentations" on the given path and a subdirectory that follows the \
pattern: YearMonthDayHourMinuteSecond. 

Outputs: 

1. Segmented images following the checkboxes on the page. \
2. stats.csv file with the statistics for each particle detected. \
3. mean_stats.csv file with the mean values and std for the detections on each image.

## Export Datasette

We indroduce a method to organise the TEM data with [Datasette](https://datasette.io/). Information can be extracted \
and stored as interactive table.

1. Export: reads a directory with .dm3 and .dm4 files and extract information from metadata. Creates a .csv file and in the sequence, \
creates a .db file. Then starts process, the adress to visualise can be found on the right hand side of the screen. Please note it is \
running locally.

2. Export from CSV: reads a .csv file (e.g. output from Segment Images page) and
creates a .db file. Then starts process, the adress to visualise can be found on the right hand side of the screen. Please note it is \
running locally.

Only one process can be running at a time. If you want to create another datasette application, press "Stop Process" first.
You can run "datasette /path/to/dbfile.db" on terminal after a .db file is created to open the application on browser.

"""