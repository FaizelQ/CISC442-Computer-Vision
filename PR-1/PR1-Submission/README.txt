Libraries
import cv2
import numpy as np
import logging as log
import yaml

- Deliverables
Q1-convolve-sobel-lena.png -> Question 1 Convolve
Q2-reduce-lena.png -> Question 2 Reduce
Q3-expand-function-lena.png -> Question 3 Expand
Q4-gaussian-pyramid-level{i}.png -> Question 4 Gaussian Pyramind
Q5-laplacian-pyramid-level{i}.png -> Question 5 Laplacian Pyramid
Q6-reconstructed-lena.png -> Question 6 Reconstruction
Q7-{A, B, C, D}.png -> Question 7 Mosaicing

- How to run mosaic?
Please run main.py and wait for image window to pop up. Please left click one point on first (left) image followed by a keypress. Please left click one point
on second (right) image followed by a key press. If not prompted back to IDE or file explorer, please do another keypress to close cv2 windows.

pr1FunctionsWorking.py
- Contains image processing functions such as convolution, reducing, expanding, Gaussian pyramind, Laplacian pyramind, and reconstruction.

pr1FunctionsInProgress.py
- Contains my earlier original implementation of these functions

main.py
- Contains testing and function calls

config.yaml
- Please load image paths into this YAML configuration