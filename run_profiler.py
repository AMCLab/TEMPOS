import cProfile
import pstats
from generator import SyntheticGenerator  # Import your entry function
import datetime
profiler = cProfile.Profile()
profiler.enable()

setting  = {
                "0": {
                    "save": True,
                    "plot_mask": False,
                    "train_val_ratio": 0,
                    "img_size":2048,
                    "grid": {
                    "type": "continuous carbon",
                    "path": None,
                    "stepSize": [
                        1024
                    ],
                    "windowSize": [
                        [
                        1024,
                        1024
                        ]
                    ],
                    "Restrict2carbon": False,
                    "Restrict2Holes": False
                    },
                    "particle": "ellipse",
                    "initial_i_min": 2,
                    "initial_i_max": 3,
                    "axes_ratio": 0.9,
                    "no_of_magnifications": 1,
                    "particle_size_min": 100,
                    "particle_size_max": 200,
                    "generate_cluster": True,
                    "cluster_type": "random",
                    "num_rows_min": 2,
                    "num_rows_max": 7,
                    "num_cols_min": 2,
                    "num_cols_max": 7,
                    "num_width_min": 2,
                    "num_width_max": 7,
                    "cluster_random_num": 100,
                    "num_of_points": 50,
                    "max_overlap": 0.5,
                    "vertices_min": 3,
                    "vertices_max": 6,
                    "irregularity": 0.2,
                    "particle_weight": 2.0,
                    "particle_noise": {
                    "type": "simplexfracta",
                    "frequency": 0.01,
                    "octaves": 5,
                    "gain": 0.5,
                    "lacunarity": 10.0
                    },
                    "particle_noise_weight": 3.0,
                    "gradient_weight": 2.0,
                    "illumination_type": "no aberration",
                    "illumination_weight": 2.0,
                    "illumination_kernel": 21,
                    "illumination_sigmaX": 2,
                    "rescale_intensity": True,
                    "rescale_intensity_prob": 1.0,
                    "brighten":False,
                    "brighten_rate":0,
                    "class_name": "particle",
                    "image_noise": {
                    "type": "perlin",
                    "frequency": 0.1,
                    "octaves": 5,
                    "gain": 0.1,
                    "lacunarity": 1.0
                    },
                    "image_noise_weight": 2.0,
                    "invert_image": False
                },
                "1": {
                    "save": True,
                    "plot_mask": False,
                    "train_val_ratio": 0,
                    "img_size":2048,
                    "grid": {
                    "type": "continuous carbon",
                    "path": None,
                    "stepSize": [
                        1024
                    ],
                    "windowSize": [
                        [
                        1024,
                        1024
                        ]
                    ],
                    "Restrict2carbon": False,
                    "Restrict2Holes": False
                    },
                    "particle": "ellipse",
                    "initial_i_min": 2,
                    "initial_i_max": 3,
                    "axes_ratio": 0.9,
                    "no_of_magnifications": 1,
                    "particle_size_min": 100,
                    "particle_size_max": 150,
                    "generate_cluster": True,
                    "cluster_type": "honeycomb",
                    "num_rows_min": 7,
                    "num_rows_max": 8,
                    "num_cols_min": 7,
                    "num_cols_max": 8,
                    "num_width_min": 50,
                    "num_width_max": 80,
                    "cluster_random_num": 99,
                    "num_of_points": 6,
                    "max_overlap": 0.5,
                    "vertices_min": 3,
                    "vertices_max": 6,
                    "irregularity": 0.2,
                    "particle_weight": 2.0,
                    "particle_noise": {
                    "type": "none",
                    "frequency": 0.01,
                    "octaves": 5,
                    "gain": 0.5,
                    "lacunarity": 10.0
                    },
                    "particle_noise_weight": 3.0,
                    "gradient_weight": 2.0,
                    "illumination_type": "no aberration",
                    "illumination_weight": 2.0,
                    "illumination_kernel": 21,
                    "illumination_sigmaX": 2,
                    "rescale_intensity": True,
                    "rescale_intensity_prob": 1.0,
                    "brighten":False,
                    "brighten_rate":0,
                    "class_name": "particle",
                    "image_noise": {
                    "type": "perlin",
                    "frequency": 0.1,
                    "octaves": 5,
                    "gain": 0.1,
                    "lacunarity": 1.0
                    },
                    "image_noise_weight": 1.5,
                    "invert_image": False
                },
                "2": {
                    "save": True,
                    "plot_mask": False,
                    "train_val_ratio": 0,
                    "img_size":2048,
                    "grid": {
                    "type": "continuous carbon",
                    "path": None,
                    "stepSize": [
                        1024
                    ],
                    "windowSize": [
                        [
                        1024,
                        1024
                        ]
                    ],
                    "Restrict2carbon": False,
                    "Restrict2Holes": False
                    },
                    "particle": "rod",
                    "initial_i_min": 2,
                    "initial_i_max": 3,
                    "axes_ratio": 1.0,
                    "no_of_magnifications": 1,
                    "particle_size_min": 150,
                    "particle_size_max": 200,
                    "generate_cluster": True,
                    "cluster_type": "honeycomb",
                    "num_rows_min": 10,
                    "num_rows_max": 51,
                    "num_cols_min": 10,
                    "num_cols_max": 51,
                    "num_width_min": 5,
                    "num_width_max": 20,
                    "cluster_random_num": 99,
                    "num_of_points": 3,
                    "max_overlap": 0.2,
                    "vertices_min": 3,
                    "vertices_max": 6,
                    "irregularity": 0.2,
                    "particle_weight": 3.0,
                    "particle_noise": {
                    "type": "simplexfracta",
                    "frequency": 0.01,
                    "octaves": 5,
                    "gain": 0.5,
                    "lacunarity": 10.0
                    },
                    "particle_noise_weight": 3.0,
                    "gradient_weight": 2.0,
                    "illumination_type": "no aberration",
                    "illumination_weight": 2.0,
                    "illumination_kernel": 21,
                    "illumination_sigmaX": 2,
                    "rescale_intensity": False,
                    "rescale_intensity_prob": 1.0,
                    "brighten":False,
                    "brighten_rate":0,
                    "class_name": "particle",
                    "image_noise": {
                    "type": "perlin",
                    "frequency": 0.1,
                    "octaves": 5,
                    "gain": 0.1,
                    "lacunarity": 1.0
                    },
                    "image_noise_weight": 1.0,
                    "invert_image": False
                },
                "3": {
                    "save": True,
                    "plot_mask": False,
                    "train_val_ratio": 0,
                    "img_size":2048,
                    "grid": {
                    "type": "continuous carbon",
                    "path": None,
                    "stepSize": [
                        1024
                    ],
                    "windowSize": [
                        [
                        1024,
                        1024
                        ]
                    ],
                    "Restrict2carbon": False,
                    "Restrict2Holes": False
                    },
                    "particle": "rod",
                    "initial_i_min": 2,
                    "initial_i_max": 3,
                    "axes_ratio": 1.0,
                    "no_of_magnifications": 1,
                    "particle_size_min": 150,
                    "particle_size_max": 200,
                    "generate_cluster": True,
                    "cluster_type": "honeycomb",
                    "num_rows_min": 50,
                    "num_rows_max": 51,
                    "num_cols_min": 50,
                    "num_cols_max": 51,
                    "num_width_min": 10,
                    "num_width_max": 11,
                    "cluster_random_num": 99,
                    "num_of_points": 1,
                    "max_overlap": 1.0,
                    "vertices_min": 3,
                    "vertices_max": 6,
                    "irregularity": 0.2,
                    "particle_weight": 3.0,
                    "particle_noise": {
                    "type": "simplexfracta",
                    "frequency": 0.01,
                    "octaves": 5,
                    "gain": 0.5,
                    "lacunarity": 10.0
                    },
                    "particle_noise_weight": 3.0,
                    "gradient_weight": 2.0,
                    "illumination_type": "no aberration",
                    "illumination_weight": 2.0,
                    "illumination_kernel": 21,
                    "illumination_sigmaX": 2,
                    "rescale_intensity": False,
                    "rescale_intensity_prob": 1.0,
                    "brighten":False,
                    "brighten_rate":0,
                    "class_name": "particle",
                    "image_noise": {
                    "type": "perlin",
                    "frequency": 0.1,
                    "octaves": 5,
                    "gain": 0.1,
                    "lacunarity": 1.0
                    },
                    "image_noise_weight": 1.0,
                    "invert_image": False
                },
                "4": {
                    "save": True,
                    "plot_mask": False,
                    "train_val_ratio": 0,
                    "img_size":2048,
                    "grid": {
                    "type": "continuous carbon",
                    "path": None,
                    "stepSize": [
                        1024
                    ],
                    "windowSize": [
                        [
                        1024,
                        1024
                        ]
                    ],
                    "Restrict2carbon": False,
                    "Restrict2Holes": False
                    },
                    "particle": "polygon",
                    "initial_i_min": 2,
                    "initial_i_max": 3,
                    "axes_ratio": 1.0,
                    "no_of_magnifications": 1,
                    "particle_size_min": 100,
                    "particle_size_max": 200,
                    "generate_cluster": True,
                    "cluster_type": "honeycomb",
                    "num_rows_min": 50,
                    "num_rows_max": 51,
                    "num_cols_min": 50,
                    "num_cols_max": 51,
                    "num_width_min": 5,
                    "num_width_max": 6,
                    "cluster_random_num": 99,
                    "num_of_points": 1,
                    "max_overlap": 1.0,
                    "vertices_min": 3,
                    "vertices_max": 8,
                    "irregularity": 0.2,
                    "particle_weight": 3.0,
                    "particle_noise": {
                    "type": "simplexfracta",
                    "frequency": 0.01,
                    "octaves": 5,
                    "gain": 0.5,
                    "lacunarity": 10.0
                    },
                    "particle_noise_weight": 3.0,
                    "gradient_weight": 2.0,
                    "illumination_type": "no aberration",
                    "illumination_weight": 2.0,
                    "illumination_kernel": 21,
                    "illumination_sigmaX": 2,
                    "rescale_intensity": False,
                    "rescale_intensity_prob": 1.0,
                    "brighten":False,
                    "brighten_rate":0,
                    "class_name": "particle",
                    "image_noise": {
                    "type": "perlin",
                    "frequency": 0.1,
                    "octaves": 5,
                    "gain": 0.1,
                    "lacunarity": 1.0
                    },
                    "image_noise_weight": 1.0,
                    "invert_image": False
                },
                "5": {
                    "save": True,
                    "plot_mask": False,
                    "train_val_ratio": 0,
                    "img_size":2048,
                    "grid": {
                    "type": "continuous carbon",
                    "path": None,
                    "stepSize": [
                        1024
                    ],
                    "windowSize": [
                        [
                        1024,
                        1024
                        ]
                    ],
                    "Restrict2carbon": False,
                    "Restrict2Holes": False
                    },
                    "particle": "polygon",
                    "initial_i_min": 2,
                    "initial_i_max": 3,
                    "axes_ratio": 1.0,
                    "no_of_magnifications": 1,
                    "particle_size_min": 200,
                    "particle_size_max": 500,
                    "generate_cluster": False,
                    "cluster_type": "honeycomb",
                    "num_rows_min": 50,
                    "num_rows_max": 51,
                    "num_cols_min": 50,
                    "num_cols_max": 51,
                    "num_width_min": 5,
                    "num_width_max": 6,
                    "cluster_random_num": 99,
                    "num_of_points": 10,
                    "max_overlap": 0.5,
                    "vertices_min": 3,
                    "vertices_max": 8,
                    "irregularity": 0.2,
                    "particle_weight": 3.0,
                    "particle_noise": {
                    "type": "none",
                    "frequency": 0.01,
                    "octaves": 5,
                    "gain": 0.5,
                    "lacunarity": 10.0
                    },
                    "particle_noise_weight": 3.0,
                    "gradient_weight": 2.0,
                    "illumination_type": "no aberration",
                    "illumination_weight": 2.0,
                    "illumination_kernel": 21,
                    "illumination_sigmaX": 2,
                    "rescale_intensity": True,
                    "rescale_intensity_prob": 1.0,
                    "brighten":False,
                    "brighten_rate":0,
                    "class_name": "particle",
                    "image_noise": {
                    "type": "perlin",
                    "frequency": 0.1,
                    "octaves": 5,
                    "gain": 0.1,
                    "lacunarity": 1.0
                    },
                    "image_noise_weight": 1.0,
                    "invert_image": False
                },
                "6": {
                    "save": True,
                    "plot_mask": False,
                    "train_val_ratio": 0,
                    "img_size":2048,
                    "grid": {
                    "type": "continuous carbon",
                    "path": None,
                    "stepSize": [
                        1024
                    ],
                    "windowSize": [
                        [
                        1024,
                        1024
                        ]
                    ],
                    "Restrict2carbon": False,
                    "Restrict2Holes": False
                    },
                    "particle": "polygon",
                    "initial_i_min": 2,
                    "initial_i_max": 3,
                    "axes_ratio": 1.0,
                    "no_of_magnifications": 1,
                    "particle_size_min": 200,
                    "particle_size_max": 500,
                    "generate_cluster": True,
                    "cluster_type": "random",
                    "num_rows_min": 50,
                    "num_rows_max": 51,
                    "num_cols_min": 50,
                    "num_cols_max": 51,
                    "num_width_min": 5,
                    "num_width_max": 6,
                    "cluster_random_num": 10,
                    "num_of_points": 10,
                    "max_overlap": 0.5,
                    "vertices_min": 3,
                    "vertices_max": 8,
                    "irregularity": 0.2,
                    "particle_weight": 3.0,
                    "particle_noise": {
                    "type": "none",
                    "frequency": 0.01,
                    "octaves": 5,
                    "gain": 0.5,
                    "lacunarity": 10.0
                    },
                    "particle_noise_weight": 3.0,
                    "gradient_weight": 2.0,
                    "illumination_type": "no aberration",
                    "illumination_weight": 2.0,
                    "illumination_kernel": 21,
                    "illumination_sigmaX": 2,
                    "rescale_intensity": True,
                    "rescale_intensity_prob": 1.0,
                    "brighten":False,
                    "brighten_rate":0,
                    "class_name": "particle",
                    "image_noise": {
                    "type": "perlin",
                    "frequency": 0.1,
                    "octaves": 5,
                    "gain": 0.1,
                    "lacunarity": 1.0
                    },
                    "image_noise_weight": 1.0,
                    "invert_image": False
                },
                "7": {
                    "save": True,
                    "plot_mask": False,
                    "train_val_ratio": 0,
                    "img_size":2048,
                    "grid": {
                    "type": "continuous carbon",
                    "path": None,
                    "stepSize": [
                        1024
                    ],
                    "windowSize": [
                        [
                        1024,
                        1024
                        ]
                    ],
                    "Restrict2carbon": False,
                    "Restrict2Holes": False
                    },
                    "particle": "polygon",
                    "initial_i_min": 2,
                    "initial_i_max": 3,
                    "axes_ratio": 1.0,
                    "no_of_magnifications": 1,
                    "particle_size_min": 200,
                    "particle_size_max": 500,
                    "generate_cluster": True,
                    "cluster_type": "random",
                    "num_rows_min": 50,
                    "num_rows_max": 51,
                    "num_cols_min": 50,
                    "num_cols_max": 51,
                    "num_width_min": 5,
                    "num_width_max": 6,
                    "cluster_random_num": 10,
                    "num_of_points": 10,
                    "max_overlap": 0.5,
                    "vertices_min": 15,
                    "vertices_max": 30,
                    "irregularity": 0.5,
                    "particle_weight": 3.0,
                    "particle_noise": {
                    "type": "none",
                    "frequency": 0.01,
                    "octaves": 5,
                    "gain": 0.5,
                    "lacunarity": 10.0
                    },
                    "particle_noise_weight": 3.0,
                    "gradient_weight": 2.0,
                    "illumination_type": "no aberration",
                    "illumination_weight": 2.0,
                    "illumination_kernel": 21,
                    "illumination_sigmaX": 2,
                    "rescale_intensity": True,
                    "rescale_intensity_prob": 1.0,
                    "brighten":False,
                    "brighten_rate":0,
                    "class_name": "particle",
                    "image_noise": {
                    "type": "perlin",
                    "frequency": 0.1,
                    "octaves": 5,
                    "gain": 0.1,
                    "lacunarity": 1.0
                    },
                    "image_noise_weight": 1.0,
                    "invert_image": False
                },
                "8": {
                    "save": True,
                    "plot_mask": False,
                    "train_val_ratio": 0,
                    "img_size":2048,
                    "grid": {
                    "type": "continuous carbon",
                    "path": None,
                    "stepSize": [
                        1024
                    ],
                    "windowSize": [
                        [
                        1024,
                        1024
                        ]
                    ],
                    "Restrict2carbon": False,
                    "Restrict2Holes": False
                    },
                    "particle": "polygon",
                    "initial_i_min": 2,
                    "initial_i_max": 3,
                    "axes_ratio": 1.0,
                    "no_of_magnifications": 1,
                    "particle_size_min": 400,
                    "particle_size_max": 500,
                    "generate_cluster": True,
                    "cluster_type": "honeycomb",
                    "num_rows_min": 3,
                    "num_rows_max": 4,
                    "num_cols_min": 2,
                    "num_cols_max": 3,
                    "num_width_min": 5,
                    "num_width_max": 6,
                    "cluster_random_num": 3,
                    "num_of_points": 3,
                    "max_overlap": 0.7,
                    "vertices_min": 15,
                    "vertices_max": 30,
                    "irregularity": 0.5,
                    "particle_weight": 3.0,
                    "particle_noise": {
                    "type": "none",
                    "frequency": 0.01,
                    "octaves": 5,
                    "gain": 0.5,
                    "lacunarity": 10.0
                    },
                    "particle_noise_weight": 3.0,
                    "gradient_weight": 2.0,
                    "illumination_type": "no aberration",
                    "illumination_weight": 2.0,
                    "illumination_kernel": 21,
                    "illumination_sigmaX": 2,
                    "rescale_intensity": True,
                    "rescale_intensity_prob": 1.0,
                    "brighten":False,
                    "brighten_rate":0,
                    "class_name": "particle",
                    "image_noise": {
                    "type": "perlin",
                    "frequency": 0.1,
                    "octaves": 5,
                    "gain": 0.1,
                    "lacunarity": 1.0
                    },
                    "image_noise_weight": 2.0,
                    "invert_image": False
                },
                "9": {
                    "save": True,
                    "plot_mask": False,
                    "train_val_ratio": 0,
                    "img_size":2048,
                    "grid": {
                    "type": "continuous carbon",
                    "path": None,
                    "stepSize": [
                        1024
                    ],
                    "windowSize": [
                        [
                        1024,
                        1024
                        ]
                    ],
                    "Restrict2carbon": False,
                    "Restrict2Holes": False
                    },
                    "particle": "polygon",
                    "initial_i_min": 2,
                    "initial_i_max": 3,
                    "axes_ratio": 1.0,
                    "no_of_magnifications": 1,
                    "particle_size_min": 100,
                    "particle_size_max": 200,
                    "generate_cluster": True,
                    "cluster_type": "honeycomb",
                    "num_rows_min": 50,
                    "num_rows_max": 60,
                    "num_cols_min": 50,
                    "num_cols_max": 60,
                    "num_width_min": 2,
                    "num_width_max": 3,
                    "cluster_random_num": 3,
                    "num_of_points": 1,
                    "max_overlap": 1.0,
                    "vertices_min": 8,
                    "vertices_max": 9,
                    "irregularity": 0.2,
                    "particle_weight": 3.0,
                    "particle_noise": {
                    "type": "none",
                    "frequency": 0.01,
                    "octaves": 5,
                    "gain": 0.5,
                    "lacunarity": 10.0
                    },
                    "particle_noise_weight": 3.0,
                    "gradient_weight": 2.0,
                    "illumination_type": "no aberration",
                    "illumination_weight": 2.0,
                    "illumination_kernel": 21,
                    "illumination_sigmaX": 2,
                    "rescale_intensity": False,
                    "rescale_intensity_prob": 1.0,
                    "brighten":False,
                    "brighten_rate":0,
                    "class_name": "particle",
                    "image_noise": {
                    "type": "perlin",
                    "frequency": 0.1,
                    "octaves": 5,
                    "gain": 0.1,
                    "lacunarity": 1.0
                    },
                    "image_noise_weight": 2.0,
                    "invert_image": False
                },
                "10": {
                    "save": True,
                    "plot_mask": False,
                    "train_val_ratio": 0,
                    "img_size":2048,
                    "grid": {
                    "type": "continuous carbon",
                    "path": None,
                    "stepSize": [
                        1024
                    ],
                    "windowSize": [
                        [
                        1024,
                        1024
                        ]
                    ],
                    "Restrict2carbon": False,
                    "Restrict2Holes": False
                    },
                    "particle": "polygon",
                    "initial_i_min": 2,
                    "initial_i_max": 3,
                    "axes_ratio": 1.0,
                    "no_of_magnifications": 1,
                    "particle_size_min": 400,
                    "particle_size_max": 450,
                    "generate_cluster": True,
                    "cluster_type": "honeycomb",
                    "num_rows_min": 7,
                    "num_rows_max": 9,
                    "num_cols_min": 7,
                    "num_cols_max": 9,
                    "num_width_min": 170,
                    "num_width_max": 190,
                    "cluster_random_num": 3,
                    "num_of_points": 1,
                    "max_overlap": 1.0,
                    "vertices_min": 8,
                    "vertices_max": 9,
                    "irregularity": 0.1,
                    "particle_weight": 3.0,
                    "particle_noise": {
                    "type": "none",
                    "frequency": 0.01,
                    "octaves": 5,
                    "gain": 0.5,
                    "lacunarity": 10.0
                    },
                    "particle_noise_weight": 3.0,
                    "gradient_weight": 2.0,
                    "illumination_type": "no aberration",
                    "illumination_weight": 2.0,
                    "illumination_kernel": 21,
                    "illumination_sigmaX": 2,
                    "rescale_intensity": True,
                    "rescale_intensity_prob": 1.0,
                    "brighten":False,
                    "brighten_rate":0,
                    "class_name": "particle",
                    "image_noise": {
                    "type": "perlin",
                    "frequency": 0.1,
                    "octaves": 5,
                    "gain": 0.1,
                    "lacunarity": 1.0
                    },
                    "image_noise_weight": 2.0,
                    "invert_image": False
                }
                }
data_train = {}
data_val = {}

data_train_coco = {'info':{
                        "description": 'Synthetic data with experimental background',
                        "year": '2024',
                        "contributor": "https://github.com/nataliaasaa",
                        "date_created": str(datetime.datetime.now())
                        },
                'images':[],
                'annotations':[],
                'categories':[
                    {"supercategory": "particle",
                    "id": 1,
                    "name": "particle"}]
                }
data_val_coco = {'info':{
                        "description": 'Synthetic data with experimental background',
                        "year": '2024',
                        "contributor": "https://github.com/nataliaasaa",
                        "date_created": str(datetime.datetime.now())
                        },
                'images':[],
                'annotations':[],
                'categories':[
                    {"supercategory": "particle",
                    "id": 1,
                    "name": "particle"}]
                }

for i in range(11):
    setting[str(i)]['binary_mask'] = True  # Enable binary mask generation

SyntheticGenerator().generate_synthetic_data(150, settings=setting, 
                                count=0, folder='./', data_train=data_train, data_val=data_val, 
                                data_train_coco=data_train_coco, data_val_coco=data_val_coco)  # Execute data generation

profiler.disable()
stats = pstats.Stats(profiler)

# Save operation counts
stats.dump_stats("operation_counts_gold.prof")

# Generate human-readable report
# with open("operation_report.txt", "w") as f:
#     stats.sort_stats("calls").print_stats(20)  # Top 20 by call count

with open("operation_report_gold.txt", "w") as f:
    stats = pstats.Stats(profiler, stream=f)
    stats.strip_dirs()  # Clean up file paths
    stats.sort_stats("cumulative")  # Sort by total time spent
    stats.print_stats(20)  # Print top 20 functions