#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import solaris as sol
import yaml

from utility import use_project_path


def create_train_csv(train_set, inference_set):
    """
    Function to create the train CSV files for solaris.
    Options for train_set and inference_set include:
    sample, tier1, tier2, test
    The set names should be passed as a string with quotes
    """

    file_array = np.rot90(np.array([
        [filename for filename in glob.iglob('temp_data/%s/tile-256/**/*.png' % train_set, recursive=True)],
        [filename for filename in glob.iglob('temp_data/%s/mask-256/**/*.png' % train_set, recursive=True)],
    ]))

    training_df = pd.DataFrame(file_array, columns=['image', 'label'])
    training_df.to_csv('temp_data/solaris_training_file.csv', index=False)

    file_array = np.rot90(np.array([
        [filename for filename in glob.iglob('temp_data/%s/**/*.png' % train_set, recursive=True)]
    ]))
    testing_df = pd.DataFrame(file_array, columns=['image'])
    testing_df.to_csv('temp_data/solaris_inference_file.csv', index=False)


def get_tile_mean(img_dir):
    """ function to calculate mean of images for preprocessing with yml file """

    # Access all PNG files in directory
    imlist = [image_file for image_file in glob.iglob('%s/**/*.png' % img_dir, recursive=True)]

    # Get the sum and count of pixel values across all images
    total_sum = 0
    total_count = 0
    chan1 = []
    chan2 = []
    chan3 = []
    chan4 = []
    for im in imlist:
        cur_img = mpimg.imread(im)
        cur_sum = np.sum(cur_img, axis=tuple(range(cur_img.ndim - 1)))
        cur_count = np.shape(cur_img)[0] * np.shape(cur_img)[1]
        total_sum += cur_sum
        total_count += cur_count

        # appending all values for std dev (might need to rework this for large sample)
        chan1.append(cur_img[0][0][0])
        chan2.append(cur_img[0][0][1])
        chan3.append(cur_img[0][0][2])
        chan4.append(cur_img[0][0][3])

    # Return the mean for images
    return (total_sum / total_count, [np.std(chan1), np.std(chan2), np.std(chan3), np.std(chan4)])


if __name__ == '__main__':
    use_project_path()
    train_collection = 'sample'
    test_collection = 'sample/tile-256'

    # Calculate and display the image mean - skipping for now and trusting the config
    # image_path = 'temp_data/%s/tile-256' % train_collection
    # avg_pix, sd_pix = get_tile_mean(image_path)
    # print(avg_pix)
    # print(sd_pix)

    # Create the csv files for training and making predictions
    create_train_csv(train_collection, test_collection)

    # Parse the yaml file into a dictionary
    config = sol.utils.config.parse('config/xdxd_spacenet4.yml')

    # Create the trainer and then kick off the training according to the config file settings
    # skip this if not training
    trainer = sol.nets.train.Trainer(config)
    trainer.train()
