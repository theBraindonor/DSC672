#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


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


class RunningStats(object):
    """
    Quick Running Mean and Standard Deviation for R,G,B pixels implementing Welford's algorithm
    """
    def __init__(self):
        """
        Initialize everything to start the running stats
        """
        self.count = 0
        self.r_mean = 0
        self.r_m2 = 0
        self.r_variance = float('nan')
        self.g_mean = 0
        self.g_m2 = 0
        self.g_variance = float('nan')
        self.b_mean = 0
        self.b_m2 = 0
        self.b_variance = float('nan')

    def update(self, r, g, b):
        """
        Update the running stats for each r,g,b value
        :param r:
        :param g:
        :param b:
        :return:
        """
        self.count += 1

        delta = r - self.r_mean
        self.r_mean += delta / self.count
        delta2 = r - self.r_mean
        self.r_m2 += delta * delta2

        delta = g - self.g_mean
        self.g_mean += delta / self.count
        delta2 = g - self.g_mean
        self.g_m2 += delta * delta2

        delta = b - self.b_mean
        self.b_mean += delta / self.count
        delta2 = b - self.b_mean
        self.b_m2 += delta * delta2

    def finalize(self):
        """
        Finalize the running stats
        :return:
        """
        self.r_variance = self.r_m2 / self.count
        self.g_variance = self.g_m2 / self.count
        self.b_variance = self.b_m2 / self.count

    def to_pandas(self):
        """
        Return the summary as a pandas data frame.  Please note this data frame returns standard deviation and
        not the variance.
        :return:
        """
        return pd.DataFrame([[
            self.count, self.r_mean, np.sqrt(self.r_variance), self.g_mean, np.sqrt(self.g_variance),
            self.b_mean, np.sqrt(self.b_variance)
        ]], columns=['count', 'r_mean', 'r_sd', 'g_mean', 'g_sd', 'b_mean', 'b_sd'])
