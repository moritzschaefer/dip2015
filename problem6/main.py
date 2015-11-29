#!/usr/bin/env python3

__author__ = 'Moritz Schäfer'
__email__ = 'mail@moritzs.de'
__sjtu_student_id = '713030990015'

'''
Requires python packages: matplotlib, scipy, numpy, pillow, cairocffi
'''


import cmath
import logging
import itertools
import sys
import math

from scipy import misc
import numpy as np
import matplotlib.pyplot as plt


def myimsave(name, arr, format=None, mode=None):
    '''
    mode : tuple/list of bools: (cut_lt_0, cut_gt_255, scale_up_values)
        Indicates how arr should be transformed if it's not uint8.
        default behaviour is (False, False, True)
        cut_lt_0 and cut_gt_255 define if values greater/less than 0 and 255 should be cut (i.e. set to 0 and 255 respectively) or scaled. scale_up_values defines if an image should be scaled to use the full valueband (0-255) if it's in a small one (e.g. 10-200).
    '''

    # Need a copy because of potential manipulation
    tmparr = arr.copy()

    if mode:
        cut_lt_0, cut_gt_255, scale_up_values = mode
    else:
        cut_lt_0, cut_gt_255, scale_up_values = (False, False, True)

    if cut_lt_0:
        tmparr[arr<0] = 0
    if cut_gt_255:
        tmparr[arr>255] = 255

    high = 255
    low = 0
    cmin = np.amin(tmparr)
    cmax = np.amax(tmparr)
    if not scale_up_values:
        if cmin > 0:
            low = cmin

        if cmax < 255:
            high = cmax

    im = misc.toimage(tmparr, high=high, low=low, cmax=cmax, cmin=cmin)
    if format is None:
        im.save(name)
    else:
        im.save(name, format)
    return


misc.imsave = myimsave


def transform_pixel_pos(pos, matrix):
    return np.dot(matrix, pos)

def interpolate(img, raw_indices, method):
    '''
    Returns new pixel value by interpolating the given pixel position with the given method over the given image
    '''
    if method == 'nearest':
        tmp_ind = np.round(raw_indices-np.finfo(np.float32).eps)

        if (tmp_ind < 0).any() or (tmp_ind >= img.shape).any():
            return 0

        return img[tmp_ind[0]][tmp_ind[1]]
    elif method == 'bilinear':
        y1, x1 = np.floor(raw_indices)
        y2, x2 = np.ceil(raw_indices)



        A = np.array([[1, y1, x1, y1*x1],
                     [1, y1, x2, y1*x2],
                     [1, y2, x1, y2*x1],
                     [1, y2, x2, y2*x2]])
        Q_indices = [(y1,x1), (y1,x2), (y2,x1), (y2,x2)]
        Qs = [img[y][x] if y >= 0 and x >= 0 and y < img.shape[0] and x < img.shape[1] else 0 for y,x in Q_indices]

        try:
            linalg_x = np.linalg.solve(A, np.array(Qs))
        except Exception as e:
            if (np.matrix(Qs) == Qs[0]).all():
                return Qs[0]
            else:
                print('Error solving linear equation. Returning 0. indices:{},{}. Values: {}, {} {}, {}'.format(*(list(raw_indices)+list(Qs))))
                return 0

        return round(np.dot(linalg_x, [1, raw_indices[0], raw_indices[1], raw_indices[0]*raw_indices[1]]))
    else:
        raise ValueError('method {} unknown'.format(method))

def rotate(img, alpha, method):
    '''
    Rotate image around image center point by given radians. Unknown areas will be black
    :param img: The image to rotate
    :param alpha: Angle to rotate in radians
    :param method: Interpolation method. One of nearest and bilinear
    :returns: Rotated image
    '''

    out = np.zeros(img.shape)
    rotation_matrix = np.array([[math.cos(alpha), math.sin(alpha)],
                                [-math.sin(alpha), math.cos(alpha)]])

    for (y,x), _ in np.ndenumerate(out):
        indices = np.array([y,x], dtype='float64')
        indices += 0.5 # Center image pixels
        rotated = transform_pixel_pos(indices - np.array(img.shape)/2, rotation_matrix)
        new_index = rotated + np.array(img.shape)/2 - 0.5
        out[y,x] = interpolate(img, new_index, method)

    return out


def translate(img, movement_vec, method='nearest'):
    '''
    Translate image by movement_vec. Undefined areas will be black. Interpolation can be used if image is moved for non-integer lengths. Though this is not recommended.
    :param img: The image to translate
    :param movement_vec: Angle to rotate in radians
    :param method: Interpolation method. One of nearest and bilinear
    :returns: Translated image
    '''
    out = np.zeros(np.round(img.shape+np.abs(movement_vec)))
    movement_vec = np.array(movement_vec)

    for (y,x), _ in np.ndenumerate(out):
        indices = np.array((y,x))
        new_indices = indices-movement_vec
        out[y][x] = interpolate(img, new_indices, method)
    return out


def scale(img, new_shape, method='nearest'):
    '''
    Return new scaled image
    :param img: Input img
    :param new_shape: Shape of the outcome image
    :param method: Interpolation method. One of nearest and bilinear
    :returns: Scaled image
    '''

    out = np.zeros(new_shape)
    scale_matrix = np.array([[(new_shape[0])/(img.shape[0]), 0],
                             [0, (new_shape[1])/(img.shape[1])]] )

    for (y,x), _ in np.ndenumerate(out):
        pixel_index = np.array((y,x))
        pixel_pos = pixel_index+0.5
        raw_indices = transform_pixel_pos(pixel_pos, scale_matrix)-0.5
        out[y][x] = interpolate(img, raw_indices, method)

    return out

def main():
    input_image_depth = 256
    input_image = misc.lena()
    misc.imsave('lena.png', input_image)
    shape = input_image.shape

    rotated = rotate(input_image, 0.4, 'nearest')
    misc.imsave('rotate_nearest.png', rotated)
    rotated = rotate(input_image, 0.4, 'bilinear')
    misc.imsave('rotate_bilinear.png', rotated)

    # move by 0.5 pixel with bilinear (note light blurring of image)
    moved = translate(input_image, (0.5,0.5), 'bilinear')
    misc.imsave('translated_bilinear.png', moved)

    # combined
    comb = translate(input_image, (100, 100), 'bilinear')
    comb = translate(comb, (-50, -50), 'nearest')
    comb = rotate(comb, 1, 'bilinear')
    comb = scale(comb, (400,400), 'bilinear')
    misc.imsave('combined.png', comb)

if __name__ == '__main__':
    main()
