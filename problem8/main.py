#!/usr/bin/env python3

__author__ = 'Moritz Schäfer'
__email__ = 'mail@moritzs.de'
__sjtu_student_id = '713030990015'

'''
Requires python packages: matplotlib, scipy, numpy, pillow, cairocffi
'''


import itertools
from enum import Enum

from matplotlib.image import imread, imsave
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

fig = plt.figure(figsize=(8,11), facecolor = 'white')

def draw_img(image, title, pos):
    ax = fig.add_subplot(3,2, pos)
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

def main():
    mask = np.array([
        [0,1,0],
        [1,1,1],
        [0,1,0],
        #[1,1,1,1,1],
        #[0,1,1,1,0],
    ])
    input_image = imread('noisy_fingerprint.tif')
    bw_image = np.zeros(input_image.shape[0:2], dtype=int)
    for y, row in enumerate(input_image):
        for x, cell in enumerate(row):
            if sum(cell[0:3]) == 0:
                bw_image[y,x] = 0
            else:
                bw_image[y,x] = 1

    imsave('orig.tif', bw_image)

    erosed_image = apply_mask(bw_image, mask, MorphType.erosion)
    dilated_image = apply_mask(bw_image, mask, MorphType.dilation)
    opened_image = opening(bw_image, mask)
    closed_image = closing(bw_image, mask)


    draw_img(bw_image, 'orig', 1)
    draw_img(erosed_image, 'erosed', 2)
    draw_img(dilated_image, 'dilated', 3)
    draw_img(opened_image, 'opened', 4)
    draw_img(closed_image, 'closed', 5)

    plt.show()




if __name__ == '__main__':
    main()
