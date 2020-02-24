#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import re

import numpy as np
import pandas as pd
from skimage import img_as_float
from skimage.color import rgba2rgb
from skimage.color import rgb2hsv
from skimage.io import imread
from sklearn.model_selection import train_test_split

from utility import use_project_path
from utility import RunningStats
from utility import write_histogram

# Used to extract the metadata for a given tile from the file path.
FILENAME_REGEX = re.compile(r'.*/([A-Za-z0-9\-_]+)/([a-zA-z]+)-([0-9]+)/([a-zA-Z0-9]+)_([a-zA-Z0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)')


def process_tile(tile_data, running_stats, tile_filename):
    """
    This function will read in the tile data and generate the needed summary and aggregate statistics based on the
    image data of the tile
    :param tile_data:
    :param running_stats:
    :param tile_filename:
    :return:
    """

    matches = FILENAME_REGEX.match(tile_filename)
    if matches:
        file_catalog = matches.group(1)
        file_type = matches.group(2)
        file_size = matches.group(3)
        file_collection = matches.group(4)
        file_map = matches.group(5)
        file_tile = matches.group(6)
        file_x = matches.group(7)
        file_y = matches.group(8)
        file_zoom = matches.group(9)

        # The Tile ID gives us a unique identifier corresponding to the tile on the map
        tile_id = '%s_%s_%s_%s' % (file_catalog, file_collection, file_map, file_tile)

        # Upsert the tile into the data source.
        tile = tile_data.get(tile_id, dict())
        tile_data[tile_id] = tile

        # Write all fo the details based on the file path.
        tile['tile_id'] = tile_id
        tile['catalog'] = file_catalog
        tile['size'] = file_size
        tile['collection'] = file_collection
        tile['map'] = file_map
        tile['tile'] = file_tile
        tile['x'] = file_x
        tile['y'] = file_y
        tile['zoom'] = file_zoom

        # If the file type is 'mask', then we are dealing with a target footprint
        if file_type == 'mask':
            mask_image = imread(tile_filename)

            # All white pixels are building footprints, so we will count them
            building_pixels = np.sum(mask_image > 0)
            tile['building_pixels'] = building_pixels

        # If the file type is 'tile', then we are dealing with a map tile
        if file_type == 'tile':
            tile_image = img_as_float(imread(tile_filename))

            # All pixels with an non-zero alpha are part of the map, so we will count them
            map_pixels = np.sum(tile_image[:, :, 3] > 0.)
            tile['map_pixels'] = map_pixels

            # We are only providing summary data on the non-black pixels, so we create an array
            # to hold them.  We will write all pixels into this array.
            rgba_pixels = np.zeros((1, map_pixels, 4))
            counter = 0
            for i in range(tile_image.shape[0]):
                for j in range(tile_image.shape[1]):
                    pixel = tile_image[i][j]
                    if pixel[3] > 0:
                        running_stats.update(pixel[0], pixel[1], pixel[2])
                        rgba_pixels[0][counter] = pixel
                        counter += 1

            # Transform the RGBA pixels into RGB pixels and then perform a histogram on each band.
            rgb_pixels = rgba2rgb(rgba_pixels)
            r_hist = np.histogram(rgb_pixels[:, :, 0], bins=32, range=(0, 1))[0]
            g_hist = np.histogram(rgb_pixels[:, :, 1], bins=32, range=(0, 1))[0]
            b_hist = np.histogram(rgb_pixels[:, :, 2], bins=32, range=(0, 1))[0]
            write_histogram(tile, r_hist, 'r')
            write_histogram(tile, g_hist, 'g')
            write_histogram(tile, b_hist, 'b')

            # Transform the RGB pixels into HSV pixels and then perform a histogram on each band.
            hsv_pixels = rgb2hsv(rgb_pixels)
            h_hist = np.histogram(hsv_pixels[:, :, 0], bins=32, range=(0, 1))[0]
            s_hist = np.histogram(hsv_pixels[:, :, 1], bins=32, range=(0, 1))[0]
            v_hist = np.histogram(hsv_pixels[:, :, 2], bins=32, range=(0, 1))[0]
            write_histogram(tile, h_hist, 'h')
            write_histogram(tile, s_hist, 's')
            write_histogram(tile, v_hist, 'v')


if  __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--collections', default='sample,sample_lg',
                        help='Collections to sample from.')
    parser.add_argument('-cp', '--collection-path', default='temp_data',
                        help='Folder containing collections.')
    parser.add_argument('-op', '--output-path', default='temp_data',
                        help='Folder to contain output.')
    parser.add_argument('-p', '--prefix', default='data',
                        help='Prefix for all data files.')
    parser.add_argument('-n', '--number', default=120,
                        help='Number of records to sample in total.')
    parser.add_argument('-s', '--split', default=0.2,
                        help='Training and testing split ratio.')
    parser.add_argument('-a', '--analyze', default='Y',
                        help='Analyze the images, Y/N.')
    parser.add_argument('-rs', '--random-state', default=42,
                        help='Random state.')
    arguments = vars(parser.parse_args())

    collections = arguments['collections']
    collection_path = arguments['collection_path']
    output_path = arguments['output_path']
    prefix = arguments['prefix']
    sample_number = int(arguments['number'])
    split_ratio = float(arguments['split'])
    analyze_images = arguments['analyze']
    random_state = int(arguments['random_state'])

    mask_rcnn_filename = '%s/%s_mask_rcnn_training_testing.csv' % (output_path, prefix)
    solaris_training_file = '%s/%s_solaris_training_file.csv' % (output_path, prefix)
    solaris_inference_file = '%s/%s_solaris_inference_file.csv' % (output_path, prefix)
    aggregation_file = '%s/%s_image_aggregation.csv' % (output_path, prefix)
    summary_file = '%s/%s_image_summary.csv' % (output_path, prefix)

    print('')
    print('Starting Creation of Training and Testing Split...')
    print('')
    print('Parameters:')
    print('        Collections: %s' % collections)
    print('    Collection Path: %s' % collection_path)
    print('        Output Path: %s' % output_path)
    print('             Prefix: %s' % prefix)
    print('      Total Samples: %s' % sample_number)
    print('        Split Ratio: %s' % split_ratio)
    print('     Analyze Images: %s' % analyze_images)
    print('       Random State: %s' % random_state)
    print('')
    print('Outputs:')
    print('      Mask R-CNN Data File: %s' % mask_rcnn_filename)
    print('     Solaris Training File: %s' % solaris_training_file)
    print('    Solaris Inference File: %s' % solaris_inference_file)
    if analyze_images == 'Y':
        print('    Image Aggregation File: %s' % aggregation_file)
        print('        Image Summary File: %s' % summary_file)
    print('')

    use_project_path()

    #
    # Step 1: Collect all tile and mask pairs in the collections into a pandas data frame.
    #
    print('Reading Collections...')
    image_mask_pairs_df = None
    for collection in collections.split(','):
        # We are reading in the files using path globbing, and remembering to do replacement for different
        # operating systems.s
        file_array = np.rot90(np.array([
            [filename.replace('\\', '/') for filename in glob.iglob('%s/%s/tile-256/**/*.png' %
                                                 (collection_path, collection), recursive=True)],
            [filename.replace('\\', '/') for filename in glob.iglob('%s/%s/mask-256/**/*.png' %
                                                 (collection_path, collection), recursive=True)],
        ]))
        if image_mask_pairs_df is None:
            image_mask_pairs_df = pd.DataFrame(file_array, columns=['image', 'mask'])
        else:
            image_mask_pairs_df = pd.concat([
                image_mask_pairs_df,
                pd.DataFrame(file_array, columns=['image', 'mask'])
            ])
    print('Done!')
    print('')

    #
    # Step 2: Create the training and testing split datafiles
    #
    print('Creating Training/Testing Split...')
    test_size = int(sample_number*split_ratio)
    train_size = sample_number - test_size

    # Scikit-Learn train_test_split is being used for reproducibility
    training_images_df, testing_images_df = train_test_split(
        image_mask_pairs_df, train_size=train_size, test_size=test_size, random_state=random_state
    )

    print('Done!')
    print('')

    #
    # Step 3: Create the Mask R-CNN training/testing file
    #
    print('Creating Mask R-CNN Training/Testing File...')

    # The Mask R-CNN model training expects a column to indicate the validation/testing set.
    mask_rcnn_training_df = training_images_df.copy()
    mask_rcnn_training_df['validation'] = np.zeros((len(mask_rcnn_training_df),))
    mask_rcnn_testing_df = testing_images_df.copy()
    mask_rcnn_testing_df['validation'] = np.ones((len(mask_rcnn_testing_df),))
    mask_rcnn_df = pd.concat([mask_rcnn_training_df, mask_rcnn_testing_df])
    mask_rcnn_df.to_csv(mask_rcnn_filename, index=False)

    print('Done!')
    print('')

    #
    # Step 4: Create the Solaris training and inference files
    #
    print('Creating Solaris Training/Inference Files...')

    # Solaris expects training and testing to be placed in separate files
    solaris_training_df = training_images_df.copy()
    solaris_training_df.columns = ['image', 'label']
    solaris_training_df.to_csv(solaris_training_file, index=False)
    solaris_testing_df = testing_images_df.copy()
    solaris_testing_df.drop('mask', axis=1, inplace=True)
    solaris_testing_df.to_csv(solaris_inference_file, index=False)

    print('Done!')
    print('')

    if analyze_images == 'N':
        exit(1)

    #
    # Step 5: Analyzing Training and Testing Images
    #
    print('Analyzing Training and Testing Images...')

    tiles = 0
    tile_data = dict()
    running_stats = RunningStats()

    for index, row in mask_rcnn_df.iterrows():
        tiles += 1
        print('%s %s %s' % (tiles, row['image'], row['mask']))
        process_tile(tile_data, running_stats, row['image'])
        process_tile(tile_data, running_stats, row['mask'])

    tile_df = pd.DataFrame.from_dict(tile_data, orient='index')
    tile_df.to_csv(aggregation_file, index=False)

    running_stats.finalize()
    summary_df = running_stats.to_pandas()
    summary_df.to_csv(summary_file, index=False)

    print('Done!')
    print('')
