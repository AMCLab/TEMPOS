#from generator_exp_background_from_dm4 import generate_experimental_background_data
import os
os.chdir('../')
from generator.generator_synthetic import SyntheticGenerator

count = 0

settings = {
  "0": {
    "save": False,
    "plot_mask": False,
    "train_val_ratio": 0.7,
    "grid": {
      "type": "lacey carbon",
      "path": None,
      "stepSize": [
        2048
      ],
      "windowSize": [
        [
          2048,
          2048
        ]
      ],
      "Restrict2carbon": False,
      "Restrict2Holes": True
    },
    "particle": "silica",
    "initial_i_min": 5,
    "initial_i_max": 7,
    "no_of_magnifications": 1,
    "particle_size_min": 70,
    "particle_size_max": 90,
    "num_of_points": 50,
    "particle_weight": 0.2,
    "particle_noise": {
      "type": "simplexfracta",
      "frequency": 0.2,
      "octaves": 15,
      "gain": 0.5,
      "lacunarity": 2.0
    },
    "particle_noise_weight": 0.1,
    "rescale_intensity": False,
    "rescale_intensity_prob": 0.5,
    "class_name": "particle",
    "image_noise": {
      "type": "perlin",
      "frequency": 0.4,
      "octaves": 10,
      "gain": 0.5,
      "lacunarity": 1.0
    },
    "image_noise_weight": 2.0
  },
  "1": {
    "save": False,
    "plot_mask": False,
    "train_val_ratio": 0.7,
    "grid": {
      "type": "holey carbon",
      "path": None,
      "stepSize": [
        2048
      ],
      "windowSize": [
        [
          2048,
          2048
        ]
      ],
      "Restrict2carbon": True,
      "Restrict2Holes": False
    },
    "particle": "polymer",
    "initial_i_min": 8,
    "initial_i_max": 9,
    "no_of_magnifications": 1,
    "particle_size_min": 100,
    "particle_size_max": 400,
    "num_of_points": 50,
    "particle_weight": 0.07,
    "particle_noise": {
      "type": "none",
      "frequency": 0.2,
      "octaves": 15,
      "gain": 0.5,
      "lacunarity": 2.0
    },
    "particle_noise_weight": 0.1,
    "rescale_intensity": False,
    "rescale_intensity_prob": 0.5,
    "class_name": "particle",
    "image_noise": {
      "type": "perlin",
      "frequency": 0.4,
      "octaves": 10,
      "gain": 0.5,
      "lacunarity": 1.0
    },
    "image_noise_weight": 2.0
  },
  "2": {
    "save": False,
    "plot_mask": False,
    "train_val_ratio": 0.7,
    "grid": {
      "type": "continuous carbon",
      "path": None,
      "stepSize": [
        2048
      ],
      "windowSize": [
        [
          2048,
          2048
        ]
      ],
      "Restrict2carbon": True,
      "Restrict2Holes": False
    },
    "particle": "polygon",
    "initial_i_min": 8,
    "initial_i_max": 9,
    "no_of_magnifications": 1,
    "particle_size_min": 100,
    "particle_size_max": 400,
    "num_of_points": 50,
    "particle_weight": 2.0,
    "particle_noise": {
      "type": "none",
      "frequency": 0.2,
      "octaves": 15,
      "gain": 0.5,
      "lacunarity": 2.0
    },
    "particle_noise_weight": 0.1,
    "rescale_intensity": True,
    "rescale_intensity_prob": 0.5,
    "class_name": "particle",
    "image_noise": {
      "type": "perlin",
      "frequency": 0.4,
      "octaves": 10,
      "gain": 0.5,
      "lacunarity": 1.0
    },
    "image_noise_weight": 2.0
  },
  "3": {
    "save": False,
    "plot_mask": False,
    "train_val_ratio": 0.7,
    "grid": {
      "type": "continuous carbon",
      "path": None,
      "stepSize": [
        2048
      ],
      "windowSize": [
        [
          2048,
          2048
        ]
      ],
      "Restrict2carbon": True,
      "Restrict2Holes": False
    },
    "particle": "rod",
    "initial_i_min": 8,
    "initial_i_max": 9,
    "no_of_magnifications": 1,
    "particle_size_min": 100,
    "particle_size_max": 400,
    "num_of_points": 50,
    "particle_weight": 2.0,
    "particle_noise": {
      "type": "none",
      "frequency": 0.2,
      "octaves": 15,
      "gain": 0.5,
      "lacunarity": 2.0
    },
    "particle_noise_weight": 0.1,
    "rescale_intensity": True,
    "rescale_intensity_prob": 0.5,
    "class_name": "particle",
    "image_noise": {
      "type": "perlin",
      "frequency": 0.4,
      "octaves": 10,
      "gain": 0.5,
      "lacunarity": 1.0
    },
    "image_noise_weight": 2.0
  },
  "4": {
    "save": False,
    "plot_mask": False,
    "train_val_ratio": 0.7,
    "grid": {
      "type": "continuous carbon",
      "path": None,
      "stepSize": [
        2048
      ],
      "windowSize": [
        [
          2048,
          2048
        ]
      ],
      "Restrict2carbon": True,
      "Restrict2Holes": False
    },
    "particle": "gold",
    "initial_i_min": 2,
    "initial_i_max": 3,
    "no_of_magnifications": 1,
    "particle_size_min": 60,
    "particle_size_max": 150,
    "num_of_points": 10,
    "particle_weight": 0.1,
    "particle_noise": {
      "type": "none",
      "frequency": 0.2,
      "octaves": 15,
      "gain": 0.5,
      "lacunarity": 2.0
    },
    "particle_noise_weight": 0.1,
    "rescale_intensity": True,
    "rescale_intensity_prob": 0.5,
    "class_name": "particle",
    "image_noise": {
      "type": "perlin",
      "frequency": 0.4,
      "octaves": 10,
      "gain": 0.5,
      "lacunarity": 1.0
    },
    "image_noise_weight": 1.0
  },
  "5": {
    "save": False,
    "plot_mask": True,
    "train_val_ratio": 0,
    "grid": {
      "type": "experimental background",
      "path": "C:/Users/Natalia/OneDrive/Documents/TEMapp/experimental_bg",
      "stepSize": [
        2048
      ],
      "windowSize": [
        [
          2048,
          2048
        ]
      ],
      "Restrict2carbon": True,
      "Restrict2Holes": False
    },
    "particle": "polymer",
    "initial_i_min": 2,
    "initial_i_max": 3,
    "no_of_magnifications": 1,
    "particle_size_min": 200,
    "particle_size_max": 600,
    "num_of_points": 10,
    "particle_weight": 0.1,
    "particle_noise": {
      "type": "none",
      "frequency": 0.2,
      "octaves": 15,
      "gain": 0.5,
      "lacunarity": 2.0
    },
    "particle_noise_weight": 0.1,
    "rescale_intensity": False,
    "rescale_intensity_prob": 0.5,
    "class_name": "particle",
    "image_noise": {
      "type": "none",
      "frequency": 0.4,
      "octaves": 10,
      "gain": 0.5,
      "lacunarity": 1.0
    },
    "image_noise_weight": 1.0
  }
}
#data_train, data_val, data_train_coco, data_val_coco, count = generate_experimental_background_data(folder, path, data_train, data_val, data_train_coco, data_val_coco, count)
_, _, _, _, _, final_image, gt_image = SyntheticGenerator().generate_synthetic_data(1, 2048, settings, count) # generator number of iterations specified, training data

import matplotlib.pyplot as plt

plt.imshow(final_image, cmap='gray')
plt.axis('off')
plt.show()
