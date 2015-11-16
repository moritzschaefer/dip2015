
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
import gzip
import os
import pickle

from scipy import misc
#from scipy.fftpack import dct
import numpy as np
import matplotlib.pyplot as plt
import gzip


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

def compress():
    pass
# zonal mask: easy
# threshold mask. WHAT?
#http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html


# with help from  http://www.lokminglui.com/dct.pdf
def dct(input_image, inverse=False):
    if input_image.shape[0] != input_image.shape[1]:
        raise ValueError('input image must be quadratic')

    # this is done for caching
    try:
        dct.T
    except AttributeError:
        # calculate T
        dct.T = np.ndarray(input_image.shape)
        for (y,x), _ in np.ndenumerate(dct.T):
            if y == 0:
                dct.T[y,x] = 1/math.sqrt(input_image.shape[0])
            else:
                dct.T[y,x] = math.sqrt(2/input_image.shape[0])*math.cos((2*x+1)*y*math.pi/(2*input_image.shape[0]))
    if not inverse:
        return np.dot(np.dot(dct.T, input_image-128), np.transpose(dct.T))
    else:
        return np.rint(np.dot(np.dot(np.transpose(dct.T), input_image), dct.T))+128

masks = [
        np.array([
            [1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1]
            ]),
        np.array([
            [1,1,1,1,1,1,0,0],
            [1,1,1,1,1,1,0,0],
            [1,1,1,1,1,0,0,0],
            [1,1,1,1,0,0,0,0],
            [1,1,1,0,0,0,0,0],
            [1,1,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0]
            ]),
        np.array([
            [1,1,1,1,0,0,0,0],
            [1,1,1,1,0,0,0,0],
            [1,1,1,0,0,0,0,0],
            [1,1,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0]
            ]),
        np.array([
            [1,1,1,0,0,0,0,0],
            [1,1,0,0,0,0,0,0],
            [1,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0]
            ])
        ]


def main():
    input_image_depth = 256
    subimage_size = 8
    input_image = misc.imread('lenna.tif')

    shape = input_image.shape

    # check if image size match
    if input_image.shape[0] % subimage_size != 0 or input_image.shape[1] % subimage_size != 0:
        raise ValueError('Image has to have height and width of multiple of {}'.format(subimage_size))

    # Print original and lossless compressed image size
    print('Original file size: {}'.format(os.stat('lenna.tif').st_size))


    f = gzip.open('lossless_compressed.pkl.gz', 'wb')
    pickle.dump(input_image,f)
    print('Compressed file size: {}'.format(os.stat('lossless_compressed.pkl.gz').st_size))

    for mask_index, mask in enumerate(masks):
        compressed = np.ndarray(shape)
        for index in np.ndindex(shape[0]/subimage_size, shape[1]/subimage_size):
            subimage = input_image[index[0]*subimage_size:(index[0]+1)*subimage_size,index[1]*subimage_size:(index[1]+1)*subimage_size]
            transformed_subimage = dct(subimage, False)
            compressed_subimage = np.ndarray((8,8))
            for (y,x), v in np.ndenumerate(transformed_subimage):
                compressed_subimage[y,x] = int(v*mask[y][x])

            if (dct(compressed_subimage, True)-subimage).max() > 30:
                import ipdb; ipdb.set_trace()

            compressed[index[0]*subimage_size:(index[0]+1)*subimage_size,index[1]*subimage_size:(index[1]+1)*subimage_size] = compressed_subimage

        # save compressed subimage to see size after compression.
        f = gzip.open('compressed{}.pkl.gz'.format(mask_index),'wb')
        pickle.dump(compressed,f)
        f.close()
        print('Stored image with mask {}. Size: {}. (Please note, that this compression is not good because the values are not stored in the "best" order. Though, improvement is beyond the scope of this assignment.)'.format(mask_index, os.stat('compressed{}.pkl.gz'.format(mask_index)).st_size))

        # now restore image to show/save result image
        restored = np.ndarray(shape)
        for index in np.ndindex(shape[0]/subimage_size, shape[1]/subimage_size):
            subimage = compressed[index[0]*subimage_size:(index[0]+1)*subimage_size,index[1]*subimage_size:(index[1]+1)*subimage_size]
            restored_subimage = dct(subimage, True)

            restored[index[0]*subimage_size:(index[0]+1)*subimage_size,index[1]*subimage_size:(index[1]+1)*subimage_size] = restored_subimage

        misc.imsave('restored_mask_{}.tif'.format(mask_index), restored, mode=(True, True, False))
        misc.imsave('diff_mask_{}.tif'.format(mask_index), np.abs(input_image-restored))

# TODO: delete
def main2():
    M = np.ones((4,4))*5
    print(M)
    print(np.rint(dct(M)))
    print(dct(dct(M),True))

#main = main2


if __name__ == '__main__':
    main()
