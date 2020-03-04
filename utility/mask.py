#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy import ndimage


def split_mask_into_layers(mask):
    # bring in mask
    label_im, nb_labels = ndimage.label(mask)
    # Set up empty array to hold split mask images
    mask_array = []
    for i in range(nb_labels):
        # create an array which size is same as the mask but filled with
        # values that we get from the label_im.
        # If there are three masks, then the pixels are labeled
        # as 1, 2 and 3.

        mask_compare = np.full(np.shape(label_im), i + 1)

        # check equality test and have the value 1 on the location of each mask
        separate_mask = np.equal(label_im, mask_compare).astype(int)

        # replace 1 with 255 for visualization as rgb image

        separate_mask[separate_mask == 1] = 255
        # append separate mask to the mask_array
        mask_array.append(separate_mask)

    return mask_array
