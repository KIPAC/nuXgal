

import os
import numpy as np
import healpy as hp

from scipy import stats

import matplotlib.pyplot as plt

gauss_sqrt2_pdf = stats.norm(loc=0, scale=1/np.sqrt(2.)).pdf

bin_edges = np.linspace(-3.05, 3.05, 62)
bin_centers = (bin_edges[0:-1] + bin_edges[1:])/2.

pdf_1d = gauss_sqrt2_pdf(bin_centers)
pdf_1d /= pdf_1d.sum()
pdf_2d = pdf_1d * np.expand_dims(pdf_1d, -1)

grid = np.meshgrid(bin_centers, np.complex(0, 1)*bin_centers)
grid_complex = grid[0] + grid[1]

pdf_2d_flat = pdf_2d.flatten()
pdf_prod = pdf_2d_flat * np.expand_dims(pdf_2d_flat, -1)

grid_flat = grid_complex.flatten()
grid_prod = grid_flat * np.expand_dims(np.conjugate(grid_flat), -1)

cl_contrib = np.real(grid_prod)


fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8,8))

axes.set_xlabel(r'Contrib to $C_{l}$')
axes.set_ylabel(r'Prob / [0.1]')

hist_bins = np.linspace(-3., 3., 61)
hist = np.histogram(cl_contrib.flat, bins=hist_bins, weights=pdf_prod.flat)

bin_centers = (hist_bins[0:-1] + hist_bins[1:])/2.
axes.plot(bin_centers, hist[0])

#img = axes.imshow(pdf_2d, extent=extent, interpolation='none')
#cbar = plt.colorbar(img)
