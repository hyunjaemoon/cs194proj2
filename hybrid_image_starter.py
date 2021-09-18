import matplotlib.pyplot as plt
from align_image_code import align_images
import numpy as np
import skimage as sk
import skimage.io as skio
from scipy.signal import convolve2d
import cv2

# Part 2.2

# First load images

# high sf
im1 = plt.imread('./DerekPicture.jpg')/255.

# low sf
im2 = plt.imread('./nutmeg.jpg')/255

im1_gray = skio.imread('DerekPicture.jpg', as_gray=True)
im1_freq = np.log(np.abs(np.fft.fftshift(np.fft.fft2(im1_gray))))
im2_gray = skio.imread('nutmeg.jpg', as_gray=True)
im2_freq = np.log(np.abs(np.fft.fftshift(np.fft.fft2(im2_gray))))
skio.imsave("out/Derek_freq.png", im1_freq)
skio.imsave("out/nutmeg_freq.png", im2_freq)

# Next align images (this code is provided, but may be improved)
im2_aligned, im1_aligned = align_images(im2, im1)

## You will provide the code below. Sigma1 and sigma2 are arbitrary
## cutoff values for the high and low frequencies

# The actual Hybrid Image process
def hybrid_image(im1, im2, sigma1, sigma2):
    # Compute Gaussian Kernel
    gauss = cv2.getGaussianKernel(15, 8)
    G = np.outer(gauss, gauss.T)

    # Create Low Pass Filter of the first image
    lowpass = np.zeros(im1.shape)
    for i in range(im1.shape[2]):
        lowpass[:, :, i] = convolve2d(im1[:, :, i], G, 'same')

    # Create High Pass Filter of the second image
    tmp = np.zeros(im2.shape)
    for i in range(im2.shape[2]):
        tmp[:, :, i] = convolve2d(im2[:, :, i], G, 'same')
    highpass = im2 - tmp

    # Add the two filters with specific sigma ratios.
    return lowpass * sigma1 + highpass * sigma2, lowpass, highpass
hybrid, lowpass, highpass = hybrid_image(im1_aligned, im2_aligned, 0.5, 1)

# Save the images. Cut Boundaries if necessary.
skio.imsave("out/Derek_lowpass.png", lowpass[250:, :])
skio.imsave("out/nutmeg_highpass.png", highpass[250:, :])
hybrid = hybrid[250:, :]
plt.imshow(hybrid)
plt.show()
skio.imsave("out/hybrid_derek_nutmeg.png", hybrid)

"""
# Repeating the above steps in one function
def hybrid_all(filename1, filename2, sigma1=1, sigma2=1):
    im1 = plt.imread(filename1)/255.
    im2 = plt.imread(filename2)/255
    name1, name2 = filename1[:filename1.index('.')], filename2[:filename2.index('.')]
    im1_aligned, im2_aligned = align_images(im1, im2)
    hybrid, _, _ = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)
    plt.imshow(hybrid)
    plt.show()
    skio.imsave("out/hybrid_{0}_{1}.png".format(name1, name2), hybrid)

#hybrid_all('wolf.jpg', 'dog.jpg')
hybrid_all('rick.jpg', 'morty.jpg', 1, 0.5)
"""

## Compute and display Gaussian and Laplacian Pyramids
## You also need to supply this function
#N = 5 # suggested number of pyramid levels (your choice)
#pyramids(hybrid, N)
