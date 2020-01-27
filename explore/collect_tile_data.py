#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import re

import numpy as np
import pandas as pd
from skimage import img_as_float
from skimage.color import rgba2rgb
from skimage.color import rgb2hsv
from skimage.io import imread

from utility import use_project_path


def write_histogram(tile, hist, prefix):
    """
    Write a histogram to a dictionary using the <prefix><bin> naming.
    :param tile: The dictionary to write the histogram to
    :param hist: The histogram
    :param prefix: The prefix to use in naming the bin
    :return:
    """
    for i in range(len(hist)):
        tile['%s%s' % (prefix, i)] = hist[i]


if __name__ == '__main__':
    use_project_path()

    # Store all of the tile data we are summarizing
    tile_data = dict()

    # Regex decodes from the filename what the contents of the tile are
    filename_regex = re.compile(r'temp_data/([A-Za-z0-9\-_]+)/([a-zA-z]+)-([0-9]+)/([a-zA-Z0-9]+)_([a-zA-Z0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)')

    tiles = 0

    # We will recurse through all of the tile PNG files in the temp_data folder
    for tile_filename in glob.iglob('temp_data/**/*.png', recursive=True):

        # Quickly coerce everything into posix-style paths
        tile_filename = tile_filename.replace('\\', '/')

        # Provide a counter so that we can see progress
        tiles += 1
        print('Tile %s' % tiles)

        # Run the filename against the regex and then pull all of the data out of the match group
        matches = filename_regex.match(tile_filename)
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

            # Skip the sample data
            if file_catalog == 'sample':
                continue

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

    # Create a pandas data frame from our dictionary and then write to file
    tile_df = pd.DataFrame.from_dict(tile_data, orient='index')
    tile_df.to_csv('temp_data/summary.csv', index=False)

    print("DONE!")
