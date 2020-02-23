#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
from pathlib import Path
import re

from skimage import img_as_ubyte
from skimage.io import imread
from skimage.io import imsave
from skimage.transform import resize

from utility import use_project_path

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-ts', '--training-set', default='raw_source_data/test',
                        help='Folder Containing Testing Data')
    parser.add_argument('-ds', '--data-set', default='temp_data/test',
                        help='Folder Containing Image Dataset')
    parser.add_argument('-s', '--size', default=256,
                        help='Pixel size of tiles to be created')
    arguments = vars(parser.parse_args())

    training_set = arguments['training_set']
    data_set = arguments['data_set']
    tile_size = int(arguments['size'])

    print('')
    print('Starting Training Image Extraction...')
    print('')
    print('Parameters:')
    print('    Training Set: %s' % training_set)
    print('        Data Set: %s' % data_set)
    print('       Tile Size: %s' % tile_size)
    print('')

    use_project_path()

    # Regex decodes from the filename what the contents of the tile are
    filename_regex = re.compile(r'.*\/([A-Za-z0-9]+).tif')

    tiles = 0

    Path('%s/tile-%s/' % (data_set, tile_size)).mkdir(parents=True, exist_ok=True)

    # We will recurse through all of the tile PNG files in the temp_data folder
    for tile_filename in glob.iglob('%s/**/*.tif' % training_set, recursive=True):

        # Quickly coerce everything into posix-style paths
        tile_filename = tile_filename.replace('\\', '/')

        # Provide a counter so that we can see progress
        tiles += 1
        print('Tile %s' % tiles)

        # Run the filename against the regex and then pull all of the data out of the match group
        matches = filename_regex.match(tile_filename)
        if matches:
            file_map = matches.group(1)

            tile_image = imread(tile_filename, plugin='pil')[0]
            tile_image = resize(tile_image, (tile_size, tile_size))

            save_filename = '%s/tile-%s/%s.png' % (data_set, tile_size, file_map)
            print(save_filename)
            imsave(save_filename, img_as_ubyte(tile_image))

print("DONE!")
