import cv2 as cv
import numpy as np
import os

#################################################################

# Write a function Convolve (I, H). I is an image of varying size, H is a kernel of varying size.
# The output of the function should be the convolution result that is displayed.

def convolve(I, H):
    

#################################################################

# Write a function Reduce(I) that takes image I as input and outputs a copy of the image resampled
# by half the width and height of the input. Remember to Gaussian filter the image before reducing it; 
# use separable 1D Gaussian kernels.

def reduce(I):


#################################################################

# Write a function Expand(I) that takes image I as input and outputs a copy of the image expanded, 
# twice the width and height of the input.

def expand(I):


#################################################################

# Use the Reduce() function to write the GaussianPyramid(I,n) function, where n is the no. of levels.

def gaussianPyramid(I, n):


#################################################################

# Use the above functions to write LaplacianPyramids(I,n) that produces n level Laplacian pyramid of I.

def laplacianPyramid(I, n):


#################################################################

# Write the Reconstruct(LI,n) function which collapses the Laplacian pyramid LI of n levels 
# to generate the original image. Report the error in reconstruction using image difference.

def reconstruct(LI, n):


#################################################################

