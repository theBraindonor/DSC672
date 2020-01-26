#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import os
import re
import traceback

import geopandas as gpd
import numpy as np
from pystac import Catalog, STAC_IO, Collection
from rasterio.transform import from_bounds
from rio_tiler import main as rt_main
from rio_tiler.errors import TileOutsideBounds
from shapely.geometry import Polygon
from skimage.io import imsave
import solaris as sol

from utility import use_project_path


def rewrite_uri(uri):
    """Several places need to know how to rewrite URIs and get the actual file"""
    # print('Before: %s' % uri)
    uri = re.sub(r'\\', '/', uri)
    uri = re.sub(r'http://localhost([CDcd]:|)/', '', uri)
    uri = re.sub(r'[a-z0-9]+-labels.json.*/([a-z0-9]+.geojson)', r'\1', uri)
    uri = re.sub(r'[a-z0-9]+.json.*/([a-z0-9]+).tif', r'\1/\1.tif', uri)
    uri = re.sub(r'collection.json[CDcd]:.*DSC672/', '', uri)
    uri = re.sub(r'[a-z0-9]+-labels.*[a-z0-9]+-labels.json.*collection.json', 'collection.json', uri)
    # print(' After: %s' % uri)
    return uri


def fixed_read_text_method(uri):
    """We need to rewrite the URI to allow for broken logic in the library for parsing local files"""
    with open(rewrite_uri(uri)) as f:
        return f.read()


STAC_IO.read_text_method = fixed_read_text_method


def save_tile_image(tif_url, xyz, tile_size, folder='', prefix=''):
    x, y, z = xyz
    tile_image, mask_image = rt_main.tile(tif_url, x , y , z, tilesize=tile_size)
    tile_image_filename = ('%s/%s_%s_%s_%s.png' % (folder, prefix, x, y, z))
    imsave(tile_image_filename, np.moveaxis(tile_image, 0, 2), check_contrast=False)
    return(tile_image_filename)


def save_mask_image(label_polygons, tile_polygons, xyz, tile_size, folder, prefix):
    x, y, z = xyz

    # Create the geometry for the mask
    tile_transformation = from_bounds(*tile_polygons.bounds, tile_size, tile_size)
    cropped_polygons = [poly for poly in label_polygons if poly.intersects(tile_polygons)]
    cropped_polygons_df = gpd.GeoDataFrame(geometry=cropped_polygons, crs='epsg:4326')

    # Create a mask using the footprint of the geometry
    tile_mask = sol.vector.mask.df_to_px_mask(df=cropped_polygons_df,
                                              channels=['footprint'],
                                              affine_obj=tile_transformation, shape=(tile_size, tile_size),
                                              boundary_width=5, boundary_type='inner', contact_spacing=5, meters=True)

    # Save the mask
    mask_image_filename = ('%s/%s_%s_%s_%s.png' % (folder, prefix, x, y, z))
    imsave(mask_image_filename, tile_mask, check_contrast=False)
    return(mask_image_filename)


def save_area_images(area_collections, area_name='nia', area_id='825a50', label_id='825a50-labels',
                     folder='temp_data/tier1', zoom_level=19, tile_size=256, counter=0):
    print('Parsing: %s %s %s' % (area_name, area_id, label_id))

    # Open the area and label from the STAC collection
    area_item = area_collections[area_name].get_item(id=area_id)
    area_label = area_collections[area_name].get_item(id=label_id)

    # Grab the geometry file and open it using geopandas
    label_geometry_file = rewrite_uri(area_label.make_asset_hrefs_absolute().assets['labels'].href)
    label_geometry_df = gpd.read_file(label_geometry_file)

    # Grab all of the geometry and then pull out the polygon data
    all_polygons = label_geometry_df.geometry
    polygon_geom = Polygon(area_item.to_dict()['geometry']['coordinates'][0])
    polygon = gpd.GeoDataFrame(index=[0], crs=label_geometry_df.crs, geometry=[polygon_geom])

    # The polygon data is saved to disk so that we can turn it into tiles.
    Path('%s/geo/' % folder).mkdir(parents=True, exist_ok=True)
    polygon['geometry'].to_file('%s/geo/%s.geojson' % (folder, area_id), driver='GeoJSON')

    # Break the polygon into a set of tiles with the corresponding size and zoom level
    
    if os.name == 'posix':
    	command_separator = '/'
    	command_prefix = './venv/bin/'
    	command_cat = 'cat'
    else:
    	command_separator = '\\'
    	command_prefix = ''
    	command_cat = 'type'
    
    command = ('%s %s/geo/%s.geojson | ' % (command_cat, folder, area_id))
    command += ('%ssupermercado burn %s | ' % (command_prefix, zoom_level))
    command += ('%smercantile shapes | ' % command_prefix)
    command += ('%sfio collect > %s/geo/%s_%s_tiles.geojson' % (command_prefix, folder, area_id, zoom_level))
    
    os.system(command.replace('/', command_separator))

    # Read the resulting tiles file and grab the xyz
    tiles = gpd.read_file('%s/geo/%s_%s_tiles.geojson' % (folder, area_id, zoom_level))
    tiles['xyz'] = tiles.id.apply(lambda x: x.lstrip('(,)').rstrip('(,)').split(','))
    tiles['xyz'] = [[int(q) for q in p] for p in tiles['xyz']]

    # Grab the filename of the TIF image
    tif_url = rewrite_uri(area_item.make_asset_hrefs_absolute().assets['image'].href)

    print("TIF URL:", tif_url)
    print("Number of tiles:", len(tiles))

    Path('%s/tile-%s/' % (folder, tile_size)).mkdir(parents=True, exist_ok=True)
    Path('%s/mask-%s/' % (folder, tile_size)).mkdir(parents=True, exist_ok=True)

    # Step through each tile and create the image and mask
    for idx in range(len(tiles)):
        tile_polygons = tiles.iloc[idx]['geometry']
        tile_image_name = save_tile_image(tif_url, tiles.iloc[idx]['xyz'], tile_size, folder,
                                          ('tile-%s/%s_%s_%s' % (tile_size, area_name, area_id, idx)))
        mask_image_name = save_mask_image(all_polygons, tile_polygons, tiles.iloc[idx]['xyz'], tile_size, folder,
                                          ('mask-%s/%s_%s_%s' % (tile_size, area_name, area_id, idx)))

        print('%s: %s, %s' % (counter, tile_image_name, mask_image_name))
        counter += 1

    return counter


if  __name__ == '__main__':
    # TODO: Have these variables pull from command line to override defaults

    #training_set = 'sample_source_data/sample'
    #temp_folder = 'temp_data/sample'

    # Uncomment for Tier 1 Data
    training_set = 'raw_source_data/train_tier_1'
    temp_folder = 'temp_data/tier1'

    # Uncomment for Tier 2 Data
    # training_set = 'raw_source_data/train_tier_2'
    # temp_folder = 'temp_data/tier2'

    zoom_level = 19
    tile_size = 256

    # Using the project path fix to make sure IDE and command line both run the same, we open up
    # the indicated catalog.  We are sending through the URL of the file via localhost to make things
    # easier to parse--due to how poorly local files are handled by pystac.
    use_project_path()
    catalog = Catalog.from_file('http://localhost/%s/catalog.json' % training_set)

    # Extract the area touples from the collections in the catalog
    areas = []
    collections = {cols.id: cols for cols in catalog.get_children()}
    for c in collections:
        items = [x for x in collections[c].get_all_items()]
        for index, item in enumerate(items):
            if index % 2 == 0 and index + 1 < len(items):
                areas.append((c, items[index].id, items[index + 1].id))

    # Quick counter for the tiles to track progress, and the ability to skip known valid areas
    tiles = 0
    skipping_valid_areas = False

    # The whitelisted area details for debugging errors
    tier1_invalid_id = ['f883a0', '42f235', '3b20d4']
    tier2_invalid_id = ['97ce35', 'bd129c']

    # All of the areas with errors will be reported
    areas_with_errors = []

    for area in areas:
        # The purpose of skipping valid areas is to ensure that we can quickly get to the errors in the files
        # without having to spend most of the time re-parsing clean data.
        if skipping_valid_areas:
            if area[1] not in tier1_invalid_id and area[1] not in tier2_invalid_id:
                continue

        # For the area, we will save all of the images.  All exceptions listed are for known exceptions that
        # were encountered when parsing the files so that they do not completely cause a parsing to fail.
        try:
            tiles = save_area_images(collections, area[0], area[1], area[2], temp_folder, zoom_level, tile_size, tiles)
        except (ValueError, AssertionError, TileOutsideBounds) as err:
            print("Exception encountered processing %s %s" % (area[0], area[1]))
            areas_with_errors.append(area)
            traceback.print_exc()

    print("The following errors threw errors:")
    print(areas_with_errors)

    # TODO: Exploration needed for why the following pieces are failing to parse:
    #       Tier 1: ('dar', 'f883a0'), ('dar', '42f235'), ('znz', '3b20d4')
    #       Tier 2: ('dar', '97ce35'), ('mah', 'bd129c')

