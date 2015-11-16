#!/usr/bin/env python3

import itertools
from enum import Enum

from scipy.ndimage.morphology import binary_dilation
from matplotlib.image import imread, imsave
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

__author__ = 'Moritz Schäfer'
__email__ = 'mail@moritzs.de'
__sjtu_student_id = '713030990015'

'''
Requires python packages: matplotlib, scipy, numpy, pillow, cairocffi
'''


def binary_imread(path, threshold = 1):
    input_image = imread(path)
    bw_image = np.zeros(input_image.shape[0:2], dtype=int)
    for y, row in enumerate(input_image):
        for x, cell in enumerate(row):
            try:
                if sum(cell[0:3]) == 0:
                    bw_image[y,x] = 0
                else:
                    bw_image[y,x] = 1
            except IndexError:
                if cell >= threshold:
                    bw_image[y,x] = 1
                else:
                    bw_image[y,x] = 0

    return bw_image


fig = plt.figure(figsize=(8,11), facecolor = 'white')

def draw_img(image, dim, title, pos):
    ax = fig.add_subplot(dim[0], dim[1], pos)
    ax.imshow(image, cmap = cm.Greys_r)
    ax.axis('off')
    ax.set_title(title)


class MorphType(Enum):
    erosion = 0
    dilation = 1


def apply_mask(image, filter_mask, morph_type):
    '''
    Applies a filter mask to an image
    :param image: The source image to apply the filter on
    :param filter_mask: The filter_mask to apply on the image
    :param morph_type: Type of morph to do
    :returns: The transformed image
    '''

    a = int((filter_mask.shape[0]-1)/2)
    b = int((filter_mask.shape[1]-1)/2)
    filter_indices = list(itertools.product(range(-a, a+1), range(-b, b+1)))

    padded_image = np.pad(image.copy(), filter_mask.shape, 'constant',
                          constant_values=0)

    img_out = np.zeros(padded_image.shape)
    for (y,x), v in np.ndenumerate(padded_image):
        # if x == padded_image.shape[0] or y == padded_image.shape[1]: ?
        #     continue
        if morph_type == MorphType.erosion:
            pix_out = 1
        elif morph_type == MorphType.dilation:
            pix_out = 0

        for s, t in filter_indices:
            if y+s > padded_image.shape[0]-1 or y+s < 0 \
                    or x+t > padded_image.shape[1]-1 or x+t < 0:
                # Doesnt matter anyways
                if morph_type == MorphType.erosion:
                    pix_out = 0
                    break
                elif morph_type == MorphType.dilation:
                    pix_out = 1
                    break
            else:
                if morph_type == MorphType.erosion:
                    if filter_mask[s, t] == 1 and padded_image[y+s, x+t] == 0:
                        pix_out = 0
                        break
                elif morph_type == MorphType.dilation:
                    if filter_mask[s, t] == 1 and padded_image[y+s, x+t] == 1:
                        pix_out = 1
                        break

        img_out[y,x] = pix_out

    return img_out[filter_mask.shape[0]:-filter_mask.shape[0],
                   filter_mask.shape[1]:-filter_mask.shape[1]]

def opening(A, B):
    tmp = apply_mask(A, B, MorphType.erosion)
    return apply_mask(tmp, B, MorphType.dilation)

def closing(A, B):
    tmp = apply_mask(A, B, MorphType.dilation)
    return apply_mask(tmp, B, MorphType.erosion)

def boundary(A, B):
    return np.abs(A - apply_mask(A, B, MorphType.erosion))

def hole_filling(A, B):

    # first find holes
    tmp_mask = np.logical_not(A)
    tmp = np.zeros(tmp_mask.shape, bool)
    output = binary_dilation(tmp, None, -1, tmp_mask, None, 1, 0)

    output = np.logical_not(output).astype('int')


    x = apply_mask(apply_mask(output - A, B, MorphType.erosion), B, MorphType.erosion)


    old_x = A.copy()

    # now apply algoirthm
    i = 1

    while (np.sum(np.abs(old_x-x))) != 0: # if there is no change left, break
        print(i)
        old_x = x
        x = apply_mask(x, B, MorphType.dilation)
        for index, value in np.ndenumerate(A):
            if value == 1:
                x[index] = 0
        i += 1
        if i >= 9:
            break

    # add up the original image
    out_img = A.copy()
    for index, value in np.ndenumerate(x):
        if value == 1:
            out_img[index] = 1

    return out_img


def connected_components(A, B):
    # TAKES LONG!
    x = np.zeros(A.shape)
    old_x = A.copy()
    x[153, 292] = 1
    x[153, 357] = 1
    x[153, 396] = 1

    i = 1

    while (np.sum(np.abs(old_x-x))) != 0: # if there is no change left, break
        print(i)
        old_x = x
        x = apply_mask(x, B, MorphType.dilation)
        for index, value in np.ndenumerate(A):
            if value == 0:
                x[index] = 0
        i += 1
        if i >= 100:
            break
    return x



def main():
    mask = np.array([
        [0,1,0],
        [1,1,1],
        [0,1,0],
    ])
    bw_image = binary_imread('noisy_fingerprint.tif')

    erosed_image = apply_mask(bw_image, mask, MorphType.erosion)
    dilated_image = apply_mask(bw_image, mask, MorphType.dilation)
    opened_image = opening(bw_image, mask)
    closed_image = closing(bw_image, mask)


    imsave('orig.tif', bw_image)
    imsave('erosion.tif', erosed_image)
    imsave('dilation.tif', dilated_image)
    imsave('opening.tif', opened_image)
    imsave('closing.tif', closed_image)

    draw_img(bw_image, (3,2), 'orig', 1)
    draw_img(erosed_image, (3,2), 'erosed', 2)
    draw_img(dilated_image, (3,2), 'dilated', 3)
    draw_img(opened_image, (3,2), 'opened', 4)
    draw_img(closed_image, (3,2), 'closed', 5)

    plt.show()



    fig.clear()
    # now task b
    mask = np.array([
        [0,1,1,1,0],
        [1,1,1,1,1],
        [1,1,1,1,1],
        [1,1,1,1,1],
        [0,1,1,1,0],
    ])
    i = 1
    for image, function in zip(['region_filling_reflections.tif',
                                'chickenfilet_with_bones.tif',
                                'licoln_from_penny.tif'],
                               [hole_filling,
                                connected_components,
                                boundary]):
        if 'chicken' in image:
            im = binary_imread(image, threshold=195)
        else:
            im = binary_imread(image)
        processed = function(im, mask)

        filename = '{}__{}'.format(function.__name__, image)
        draw_img(processed, (1,3), filename, i)
        imsave(filename,
               processed)

        i += 1

    plt.show()

if __name__ == '__main__':
    main()
