#!/usr/bin/env python
# coding: utf-8

# In[1]:


# CS194-26 (CS294-26): Project 2 by Hyun Jae Moon

# Import all libraries necessary
import time
import numpy as np
import skimage as sk
import skimage.io as skio
from scipy.signal import convolve2d
import cv2


# In[2]:


# Part 1.1
# Read the image and conver to float
im = skio.imread('cameraman.png', as_gray=True)
im = sk.img_as_float(im)

dx = np.array(np.mat('1 -1'))
dy = np.array(np.mat('1;-1'))

# Perform Convolving Image with dx
gx = convolve2d(im, dx, 'same')

skio.imshow(gx, cmap='gray')
skio.show()
skio.imsave("out/gx.png", gx)

# Perform Convolving Image with dy
gy = convolve2d(im, dy, 'same')

skio.imshow(gy, cmap='gray')
skio.show()
skio.imsave("out/gy.png", gy)

# Compute Magnitude
mag = np.sqrt(np.square(gx) + np.square(gy))

skio.imshow(mag, cmap='gray')
skio.show()
skio.imsave("out/mag.png", mag)

# Turning it into edge image with threshold
th = 0.1
edge = mag>th
edge = edge*1
skio.imshow(edge, cmap='gray')
skio.show()
skio.imsave("out/edge.png", edge)


# In[3]:


# Part 1.2
# Read the image and conver to float
im = skio.imread('cameraman.png', as_gray=True)
im = sk.img_as_float(im)

# Compute 2D Gaussian Kernel by outer producting with a transpose
gauss = cv2.getGaussianKernel(3, 1)
G = np.outer(gauss, gauss.T)

th = 0.1 #threshold = 0.1

# convolving with gaussian to get less noise
gx_smooth = convolve2d(gx, G, 'same')
gy_smooth = convolve2d(gy, G, 'same')
mag = np.sqrt(np.square(gx_smooth) + np.square(gy_smooth))
edge = mag>=th
edge = edge*1
skio.imshow(edge, cmap='gray')
skio.show()
skio.imsave("out/gauss_edge.png", edge)

# taking partial derivatives of G first
DoGx = convolve2d(G, dx)
DoGy = convolve2d(G, dy)

# convolving with each derivatives of G
DoGx_gx = convolve2d(im, DoGx, 'same')
DoGy_gy = convolve2d(im, DoGy, 'same')
mag = np.sqrt(np.square(DoGx_gx) + np.square(DoGy_gy))
edge = mag>=th
edge = edge*1
skio.imshow(edge, cmap='gray')
skio.show()
skio.imsave("out/gauss_edge_single.png", edge)


# In[4]:


# Part 2.1

# Gaussian Convolution from Part 1.2
gauss = cv2.getGaussianKernel(12, 2)
G = np.outer(gauss, gauss.T)

# Defining Image Sharpening Function
def sharpen_image(filename, alpha=1, save=True):
    name = filename[:filename.index('.')]
    im = skio.imread(filename)
    im = sk.img_as_float(im)
    im_blur = np.zeros(im.shape) # initialize blurred image to zero
    # Since the image is in ND array, we must convolve each color channels
    for i in range(im.shape[2]):
        im_blur[:,:,i] += convolve2d(im[:,:,i], G, 'same')
    skio.imshow(im_blur)
    skio.show()
    if save:
        skio.imsave("out/{0}_alpha{1}_blur.png".format(name, alpha), im_blur)
    # Unsharp Mask Filter Formula
    sharp = im + alpha * (im - im_blur)
    skio.imshow(sharp)
    skio.show()
    # Must np.clip to remove negative values
    if save:
        skio.imsave("out/{0}_alpha{1}_sharp.png".format(name, alpha), np.clip(sharp, 0, 1))

# Sharpening the image by adding the gauss edge values (taj.jpg)
sharpen_image("taj.jpg")

# Sharpening the image by adding the gauss edge values (mario.jpg)
sharpen_image("mario.jpg")

# Sharpening the image by adding the gauss edge values (elon.jpg)
sharpen_image("elon.jpg")


# In[5]:


# elon.jpg with different alpha values

sharpen_image("elon.jpg", 1)
sharpen_image("elon.jpg", 2)
sharpen_image("elon.jpg", 3)


# In[6]:


# Part 2.3 and Part 2.4

# Gaussian Convolution from Part 1.2
gauss = cv2.getGaussianKernel(12, 2)
G = np.outer(gauss, gauss.T)

# Gaussian Stacks
im1 = skio.imread("apple.jpeg")/255.
im2 = skio.imread("orange.jpeg")/255.

# Cropping and Combining as an example
apple = im1
orange = im2
comb = np.zeros(im1.shape)
halfim1 = im1[:, :im1.shape[1]//2]
halfim2 = im2[:, im2.shape[1]//2:]
level = 6
for i in range(3):
    comb[:, :, i] = np.append(halfim1[:, :, i], halfim2[:, :, i], axis=1)
skio.imsave("out2/crop_comb.png", comb)

# A function that creates gaussian stack
def create_gauss_stack(im, name, show=False, save=True):
    gauss_stack = [im]
    for i in range(level):
        tmp = np.zeros(im.shape)
        for j in range(im.shape[2]):
            tmp[:, :, j] = convolve2d(im[:, :, j], G, 'same')
        gauss_stack.append(tmp)
        im = tmp
        if show:
            skio.imshow(tmp)
            skio.show()
    if save:
        for i in range(len(gauss_stack)):
            skio.imsave("out2/{0}_gauss{1}.png".format(name,i+1), gauss_stack[i])
    return gauss_stack

# A function that creates laplacian stack
def create_lap_stack(gauss_stack, name, show=False, save=True):
    lap_stack = []
    for i in range(level):
        tmp = gauss_stack[i+1]-gauss_stack[i]
        lap_stack.append(tmp)
        if show:
            skio.imshow(tmp)
            skio.show()
        if save:
            skio.imsave("out2/{0}_lap{1}.png".format(name,i+1), tmp)
    return lap_stack

# Reversing the laplacian stack to produce the original image.
def collapse_stack(last_gauss, lap_stack, name):
    tmp = last_gauss
    for lap in lap_stack[::-1]:
        tmp = tmp - lap
    skio.imsave("out2/{0}_collapse.png".format(name), tmp)

# Blind blending of a whole image (works greate for oraple.)
def alpha_blending(im1, im2, name):
    assert im1.shape == im2.shape
    result = np.zeros(im1.shape)
    for i in range(im1.shape[2]):
        for j in range(im1.shape[1]):
            result[:, j, i] = im1[:, j, i]*(1-j/(im1.shape[1]-1)) + im2[:, j, i]*(j/(im1.shape[1]-1))
    skio.imsave("out2/{0}_hybrid.png".format(name), result)
    return result

# The multiresolution blending
def multires_blending(mask, gauss_back, gauss_front, lap_back, lap_front, shape, name):
    blend_stack = []
    # Create a gaussian stack of a mask
    gauss_region = create_gauss_stack(mask, "mask", save=False)
    # Blend the laplacian stack at each level
    for idx in range(len(lap_back)):
        tmp = np.zeros(shape)
        for i in range(shape[2]):
            tmp[:, :, i] = gauss_region[idx][:, :, i]*lap_back[idx][:, :, i] + (1 - gauss_region[idx][:, :, i])*lap_front[idx][:, :, i]
        blend_stack.append(tmp)
    last_gauss = np.zeros(shape)
    for i in range(shape[2]):
        last_gauss[:, :, i] = gauss_region[-1][:, :, i]* gauss_back[-1][:, :, i] + (1- gauss_region[-1][:, :, i])*gauss_front[-1][:,:,i]
    # Reverse the laplacian stack to produce a blended image
    collapse_stack(last_gauss, blend_stack, name)
    return blend_stack

# Create Mask
black = np.zeros((300,150,3))
white = np.ones((300,150,3))
mask = np.hstack((black,white))

# Regular Alpha Blending of apple and orange
blend = alpha_blending(apple, orange, 'apple_orange')

# Create apple stacks
apple_gauss_stack = create_gauss_stack(apple, 'apple', save=False)
apple_lap_stack = create_lap_stack(apple_gauss_stack, 'apple', save=False)

# Create orange stacks
orange_gauss_stack = create_gauss_stack(orange, 'orange', save=False)
orange_lap_stack = create_lap_stack(orange_gauss_stack, 'orange', save=False)

# Create black stacks
black_gauss_stack = create_gauss_stack(np.zeros((300, 300, 3)), 'black', save=False)
black_lap_stack = create_lap_stack(black_gauss_stack, 'black', save=False)

# Create blend stacks and save the resulting blended image
blend_stack = multires_blending(mask, orange_gauss_stack, apple_gauss_stack, orange_lap_stack, apple_lap_stack, im1.shape, 'multires')
orange_stack = multires_blending(mask, orange_gauss_stack, black_gauss_stack, orange_lap_stack, black_lap_stack, im1.shape, 'orange_black')
apple_stack = multires_blending(mask, black_gauss_stack, apple_gauss_stack, black_lap_stack, apple_lap_stack, im1.shape, 'apple_black')

# Save each laplacian steps of each image
for i in range(5):
    skio.imsave("out2/oraple_blend{0}.png".format(i+1), blend_stack[i])
    skio.imsave("out2/apple_lap{0}.png".format(i+1), apple_stack[i])
    skio.imsave("out2/orange_lap{0}.png".format(i+1), orange_stack[i])


# In[7]:


# Part 2.4 Continued (personal examples)

# River and Lava: Possible weather in 2200
im1 = skio.imread("river.jpg")/255.
im2 = skio.imread("lava.jpg")/255.
level = 5

print(im1.shape)
print(im2.shape)

im2 = im2[20:683, 120:1120]

skio.imshow(im1)
skio.show()
skio.imshow(im2)
skio.show()

river, lava = im1, im2

river_gauss_stack = create_gauss_stack(river, 'river', save=False)
river_lap_stack = create_lap_stack(river_gauss_stack, 'river', save=False)

lava_gauss_stack = create_gauss_stack(lava, 'lava', save=False)
lava_lap_stack = create_lap_stack(lava_gauss_stack, 'lava', save=False)

black = np.zeros((130,1000,3))
white = np.ones((533,1000,3))
mask = np.vstack((black,white))

result = multires_blending(mask, lava_gauss_stack, river_gauss_stack, lava_lap_stack, river_lap_stack, river.shape, 'river_lava')


# In[8]:


# Part 2.4 Continued (personal examples)

# Mario and Luigi's Hat Dispute
mario = skio.imread("mario.png")/255.
luigi = skio.imread("luigi.png")/255.
level = 50

mario = mario[:175,45:200]
luigi = luigi[:175,45:200]

print(mario.shape)
print(luigi.shape)

skio.imshow(mario)
skio.show()
skio.imshow(luigi)
skio.show()

mask1 = np.zeros((8, 155, 4))
mask2_black = np.zeros((37, 52, 4))
mask2_white = np.ones((37, 50, 4))
mask2_rest = np.zeros((37, 53, 4))
mask2 = np.hstack((mask2_black, mask2_white, mask2_rest))
mask3 = np.zeros((130, 155, 4))
mask = np.vstack((mask1, mask2, mask3))

mario_gauss_stack = create_gauss_stack(mario, 'mario', save=False)
mario_lap_stack = create_lap_stack(mario_gauss_stack, 'mario', save=False)

luigi_gauss_stack = create_gauss_stack(luigi, 'luigi', save=False)
luigi_lap_stack = create_lap_stack(luigi_gauss_stack, 'luigi', save=False)

result = multires_blending(mask, luigi_gauss_stack, mario_gauss_stack, luigi_lap_stack, mario_lap_stack, mario.shape, 'mario_luigi')


# In[ ]:
