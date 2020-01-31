import os
import glob
import csv
import solaris as sol
import skimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def create_train_csv(train_set, inference_set):
    """
    Function to create the train CSV files for solaris.
    Options for train_set and inference_set include:
    sample, tier1, tier2, test
    The set names should be passed as a string with quotes
    """

    # !!! Optional parameters to sample from the training and inference set might be helpful for creating small tests

    # This references the project director "DSC672" as of now, but needs to be cleaned up
    project_dir = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..', '..', '..'))

    # Collect the absolute file paths for the image files
    os.chdir(project_dir + '\\temp_data\\' + train_set + '\\tile-256')
    tile_paths = []
    for dirpath,_,filenames in os.walk(os.getcwd()):
        for f in filenames:
            tile_paths.append(os.path.abspath(os.path.join(dirpath, f)))
    tile_paths.sort()
    tile_paths.insert(0,'image')

    # Collect the absolute file paths for the mask files
    os.chdir(project_dir + '\\temp_data\\' + train_set + '\\mask-256')
    mask_paths = []
    for dirpath,_,filenames in os.walk(os.getcwd()):
        for f in filenames:
            mask_paths.append(os.path.abspath(os.path.join(dirpath, f)))
    mask_paths.sort()
    mask_paths.insert(0,'label')

    # Create the csv files to the approriate temp_data subfolder
    with open(project_dir + '\\temp_data\\' + train_set + '\\training_file.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(zip(tile_paths, mask_paths))

    with open(project_dir + '\\temp_data\\' + train_set + '\\inference_file.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(zip(tile_paths))

def get_tile_mean(img_dir):
    """ function to calculate mean of images for preprocessing with yml file """

    # Change directory for desired images
    os.chdir(img_dir)

    # Access all PNG files in directory
    # !!! Need to update the standard deviation calculation for a single path, current code could cause memory error
    allfiles = os.listdir()
    imlist = [filename for filename in allfiles if filename[-4:] in [".png", ".PNG"]]

    # Get the sum and count of pixel values across all images
    total_sum = 0
    total_count = 0
    chan1 = []
    chan2 = []
    chan3 = []
    chan4 = []
    for im in imlist:
        cur_img = mpimg.imread(img_dir + '\\' + im)
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

# Calculate and display the image mean
project_path = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
image_path = project_path + '\\temp_data\\sample\\tile-256'
avg_pix, sd_pix = get_tile_mean(image_path)
print(avg_pix)
print(sd_pix)

# Create the csv files for training and making predictions
create_train_csv("sample", "sample")

# Parse the yaml file into a dictionary - will need to update this absolute path
# The yml file also contains absolute references that will need to be updated for each user
config = sol.utils.config.parse('C:\\Users\\Brian\\Documents\\GitHub\\DSC672\\temp_data\\xdxd_spacenet4.yml')

# Create the trainer and then kick off the training according to the config file settings
# skip this if not training
trainer = sol.nets.train.Trainer(config)
trainer.train()

# !!! This is where i'm a little lost on how to create predictions - This creates results but with warnings !!! #
inferer = sol.nets.infer.Inferer(config)
inf_df = sol.nets.infer.get_infer_df(config)
inferer(inf_df)

# create sample images to view results for 1 tile
actual_image_sample = skimage.io.imread('C:\\Users\\Brian\\Documents\\GitHub\\DSC672\\temp_data\\sample\\tile-256\\nia_825a50_107_265064_242184_19.png')
actual_mask_sample = skimage.io.imread('C:\\Users\\Brian\\Documents\\GitHub\\DSC672\\temp_data\\sample\\mask-256\\nia_825a50_107_265064_242184_19.png')
continuous_prediction_sample = skimage.io.imread('C:\\Users\\Brian\\Documents\\GitHub\\DSC672\\temp_data\\sample\\inference_out\\nia_825a50_107_265064_242184_19.png')

continuous_prediction_sample = continuous_prediction_sample - continuous_prediction_sample.min()
continuous_prediction_sample = continuous_prediction_sample / continuous_prediction_sample.max()

src_img_path = 'C:\\Users\\Brian\\Documents\\GitHub\\DSC672\\temp_data\\sample\\tile-256\\nia_825a50_107_265064_242184_19.png'

inferred_vectors = sol.vector.mask.mask_to_poly_geojson(
    continuous_prediction_sample,
    bg_threshold=0.80,
    reference_im=src_img_path,
    do_transform=False
)

im_arr = skimage.io.imread(src_img_path)
binary_prediction_sample = sol.vector.mask.footprint_mask(inferred_vectors,
                                          reference_im=src_img_path)

# display sample results
fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(actual_image_sample, cmap='gray')
axs[0, 0].set_title('Original Image Tile')
axs[0, 1].imshow(actual_mask_sample, cmap='gray')
axs[0, 1].set_title('Actual Image Mask')
axs[1, 0].imshow(continuous_prediction_sample, cmap='gray')
axs[1, 0].set_title('Continous Probability Prediction')
axs[1, 1].imshow(binary_prediction_sample, cmap='gray')
axs[1, 1].set_title('Binary Prediction')
plt.show()

# !!! Need to add code that that test against the competition testing data

# !!! Need to add the code for model evaluation to see how this stacks up for jaccard

# !!! Need to figure out how to pass the saved weights for inferring

# !!! Need to update file paths to be relative for team / github