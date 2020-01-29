#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
from pathlib import Path
import re

from skimage import img_as_ubyte
from skimage.io import imread
from skimage.io import imsave
from skimage.transform import resize

from utility import use_project_path

if __name__ == '__main__':
    tile_size = 256
    temp_folder = 'temp_data/test'

    use_project_path()

    # Regex decodes from the filename what the contents of the tile are
    filename_regex = re.compile(r'.*\/([A-Za-z0-9]+).tif')

    tiles = 0

    Path('%s/tile-%s/' % (temp_folder, tile_size)).mkdir(parents=True, exist_ok=True)

    # We will recurse through all of the tile PNG files in the temp_data folder
    for tile_filename in glob.iglob('raw_source_data/test/**/*.tif', recursive=True):

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

            save_filename = '%s/tile-%s/%s.png' % (temp_folder, tile_size, file_map)
            print(save_filename)
            imsave(save_filename, img_as_ubyte(tile_image))

print("DONE!")
