

import os
import numpy as np
import healpy as hp

from scipy import stats

import matplotlib.pyplot as plt

gauss_sqrt2_pdf = stats.norm.pdf

bin_edges = np.linspace(-3.05, 3.05, 602)
bin_centers = (bin_edges[0:-1] + bin_edges[1:])/2.

pdf_1d = gauss_sqrt2_pdf(bin_centers)
pdf_1d /= pdf_1d.sum()
pdf_2d = pdf_1d * np.expand_dims(pdf_1d, -1)

cl_contrib = bin_centers * np.expand_dims(bin_centers, -1)


fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8,8))

axes.set_xlabel(r'Contrib to $C_{l}$')
axes.set_ylabel(r'Prob / [0.1]')

hist_bins = np.linspace(-3., 3., 61)
hist = np.histogram(cl_contrib.flat, bins=hist_bins, weights=pdf_2d.flat)

bin_centers = (hist_bins[0:-1] + hist_bins[1:])/2.
axes.plot(bin_centers, hist[0])

#img = axes.imshow(pdf_2d, extent=extent, interpolation='none')
#cbar = plt.colorbar(img)
