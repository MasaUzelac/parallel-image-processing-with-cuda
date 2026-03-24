# CUDA Image Processing

A CUDA / PyCUDA project for parallel image processing on the GPU.

This project implements three fundamental image processing operations:
- grayscale conversion
- Gaussian blur
- brightness adjustment

Each algorithm is parallelized so that GPU threads process image pixels efficiently, allowing scalable execution on images of different dimensions.

## Features

- Parallel grayscale conversion
- Parallel Gaussian blur with configurable kernel size
- Parallel brightness adjustment based on mean intensity
- Image processing using GPU kernels
- Support for images larger than a single block
- NumPy-based image loading and preprocessing

## Algorithms

### Grayscale
Converts RGB images to grayscale using the standard luminance formula:

`0.299 * R + 0.587 * G + 0.114 * B`

### Gaussian Blur
Applies Gaussian convolution independently to image channels.

- Gaussian filter generated on CPU
- configurable kernel size
- convolution executed on GPU
- edge handling included

### Brightness Adjustment
Adjusts brightness by scaling the difference between each pixel value and the mean image intensity.

Implemented in two stages:
1. compute pixel intensity sum
2. apply scaling relative to the mean

## Tech Stack

- Python
- CUDA / PyCUDA
- NumPy
