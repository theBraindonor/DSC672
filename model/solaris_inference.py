
import solaris as sol
import numpy as np
import pandas as pd
import skimage
import matplotlib.pyplot as plt
import glob
import os
from PIL import Image

from utility import use_project_path

def create_train_csv(train_set, inference_set):
    """
    Function to create the train CSV files for solaris.
    Options for train_set and inference_set include:
    sample, tier1, tier2, test
    The set names should be passed as a string with quotes
    """

    if inference_set == 'test':

        file_array = np.rot90(np.array([
            [filename for filename in glob.iglob('temp_data/%s/tile-256/**/*.png' % train_set, recursive=True)],
            [filename for filename in glob.iglob('temp_data/%s/mask-256/**/*.png' % train_set, recursive=True)],
        ]))

        training_df = pd.DataFrame(file_array, columns=['image', 'label'])
        training_df.to_csv('temp_data/solaris_training_file.csv', index=False)

        file_array = np.rot90(np.array([
            [filename for filename in glob.iglob('temp_data/temp_test/resize/*.png', recursive=True)],
        ]))

        testing_df = pd.DataFrame(file_array, columns=['image'])
        testing_df.to_csv('temp_data/solaris_inference_file.csv', index=False)

    else:

        file_array = np.rot90(np.array([
            [filename for filename in glob.iglob('temp_data/%s/tile-256/**/*.png' % train_set, recursive=True)],
            [filename for filename in glob.iglob('temp_data/%s/mask-256/**/*.png' % train_set, recursive=True)],
        ]))

        training_df = pd.DataFrame(file_array, columns=['image', 'label'])
        training_df.to_csv('temp_data/solaris_training_file.csv', index=False)

        file_array = np.rot90(np.array([
            [filename for filename in glob.iglob('temp_data/%s/tile-256/**/*.png' % train_set, recursive=True)]
        ]))

        testing_df = pd.DataFrame(file_array, columns=['image'])
        testing_df.to_csv('temp_data/solaris_inference_file.csv', index=False)

def submission_predictions(threshold):
    raw_file_array = np.rot90(np.array([
        [filename for filename in glob.iglob('raw_source_data/test/**/*.tif', recursive=True)],
    ]))

    for file in raw_file_array:
        im = Image.open(file[0])
        im = im.resize((256, 256))
        file_base = os.path.basename(file[0])
        file_id = file_base.split('.')[0]
        resize_path = os.path.join('temp_data\\temp_test\\resize', file_id + '.png')
        im.save(resize_path)

    create_train_csv('test', 'test')

    config = sol.utils.config.parse('config/xdxd_spacenet4.yml')

    inferer = sol.nets.infer.Inferer(config)
    inf_df = sol.nets.infer.get_infer_df(config)
    inferer(inf_df)

    test_file_array = np.rot90(np.array([
        [filename for filename in glob.iglob('temp_data/temp_test/inference_out/*.png', recursive=True)],
    ]))

    for file in test_file_array:
        continuous_pred = skimage.io.imread(file[0])
        continuous_pred = continuous_pred - continuous_pred.min()
        continuous_pred = continuous_pred / continuous_pred.max()
        file_base = os.path.basename(file[0])
        file_id = file_base.split('.')[0]
        src_img_path = os.path.join('temp_data\\temp_test\\resize', file_base)
        inferred_vectors = sol.vector.mask.mask_to_poly_geojson(continuous_pred,
                                                                bg_threshold=threshold,
                                                                reference_im=src_img_path)
        binary_pred = sol.vector.mask.footprint_mask(inferred_vectors, reference_im=src_img_path)
        img = Image.fromarray(binary_pred)
        img = img.resize((1024, 1024))
        submission_file = os.path.join('temp_data\\temp_test\\submission', file_id + ".TIFF")
        img.save(submission_file)

if __name__ == '__main__':
    use_project_path()

    # run predictions for submission
    submission_predictions(0.85)