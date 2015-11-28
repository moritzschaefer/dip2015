#!/usr/bin/env python3

__author__ = 'Moritz Schäfer'
__email__ = 'mail@moritzs.de'
__sjtu_student_id = '713030990015'

'''
Requires python packages: matplotlib, scipy, numpy, pillow, cairocffi
'''

# TODO: scale some stuff (see book)...

import math
import logging
import itertools

from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

def apply_mask(image, filter_mask, scale_to = -1):
    '''
    Applies a filter mask to an image
    :param image: The source image to apply the filter on
    :param filter_mask: The filter_mask to apply on the image
    :param scale_to: If greater than zero, scale the output image to contain values only in [0,scale_to]
    :returns: The transformed image
    '''

    # From formulae 3.5-1 to apply a filter mask to an image:
    # g(x,y)=sums=-a->a(sumt=-b->b(w(s,t)f(x+s,y+t)))
    # filter is mxn, a=(m-1)/2, b=(n-1)/2; x \in [0,M-1]; y \in [0,N-1]

    a = int((filter_mask.shape[0]-1)/2)
    b = int((filter_mask.shape[1]-1)/2)
    filter_indices = list(itertools.product(range(-a,a+1), range(-b,b+1)))


    img_out = np.zeros(image.shape)
    for (x,y), v in np.ndenumerate(image):
        # if x == image.shape[0] or y == image.shape[1]:
        #     continue
        pix_out = 0
        for s,t in filter_indices:
            if x+s > image.shape[0]-1 or x+s < 0 \
                    or y+t > image.shape[1]-1 or y+t < 0:
                continue
            else:
                pix_out += image[x+s, y+t]*filter_mask[s,t]
        #filter_image_values = [image[x+s, y+t]*filter_mask[s,t] for s,t in filter_indices]

        img_out[x,y] = pix_out

    # scale image to be in range [0,256]
    if scale_to > 0:
        # some scaling..
        img_out[img_out<0] = 0
        img_out /= (np.amax(img_out)/scale_to)

    return img_out.astype(int)


def main():
    # load image. scipy includes standard lena image
    input_image_depth = 256
    input_image = misc.imread('skeleton_orig.tif')
    misc.imsave('skeleton_orig.png', input_image)

    # skeleton_laplacian ( b )
    laplacian_filter_d = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1],
        ])

    b_image = apply_mask(input_image, laplacian_filter_d, 256)
    misc.imsave('b_laplacian_filter_d.png', b_image)

    # (c) laplacian with original (same as a+b)
    laplacian_filter_d_with_original = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1],
        ])
    c_image = apply_mask(input_image, laplacian_filter_d_with_original, 256)
    misc.imsave('c_laplacian_filter_d_with_original.png', c_image)
    c_image = input_image+b_image
    misc.imsave('c2_laplacian_filter_d_with_original.png', input_image+b_image)

    # (d) sobel image
    g_x_filter = np.array([
        [-1,0,1],
        [-2,0,2],
        [-1,0,1]
        ])
    g_y_filter = np.array([
        [-1,-2,-1],
        [0,0,0],
        [1,2,1],
        ])

    g_x = apply_mask(input_image, g_x_filter)
    g_y = apply_mask(input_image, g_y_filter)

    # for faster computation just use this. d_image is the "derivative" image (not really derivative because the computation with absolutes is not correct for being fast.)
    d_image = np.absolute(g_x) + np.absolute(g_y)
    misc.imsave('d_sobel.png', d_image)

    # (e) averaged/smoothed  the sobel image (d))
    averaging_filter = np.array([
        [1/25,1/25,1/25,1/25,1/25],
        [1/25,1/25,1/25,1/25,1/25],
        [1/25,1/25,1/25,1/25,1/25],
        [1/25,1/25,1/25,1/25,1/25],
        [1/25,1/25,1/25,1/25,1/25]
        ])

    e_image = apply_mask(d_image, averaging_filter)
    misc.imsave('e_smoothed_sobel.png', e_image)

    # (f) multiply (e) and (c)


    f_image = e_image*c_image/(256)

    misc.imsave('f_derivative_1_and_2_multiplied.png', f_image.astype(int)) # f_image.astype(int)

    # (g) = (a) + (f)

    g_image = f_image+input_image
    misc.imsave('g_orig_plus_multiplied.png', g_image)

    # (h) = power law transformation
    gamma = 0.40 # values from the book
    c = 1.0

    h_image = f_image**gamma
    misc.imsave('h_g_powerlaw_transformed.png', h_image)

if __name__ == '__main__':
    main()
