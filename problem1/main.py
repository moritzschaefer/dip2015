#!/usr/bin/env python3

__author__ = 'Moritz Schäfer'
__email__ = 'mail@moritzs.de'
__sjtu_student_id = '713030990015'

'''
Requires python packages: matplotlib, scipy, numpy, pillow, cairocffi
'''


import math
import logging

from scipy import misc
import numpy as np
import matplotlib.pyplot as plt


# got this from http://stackoverflow.com/questions/6824122/mapping-a-numpy-array-in-place
def array_map(a, f):
    t = np.copy(a)
    t = t.reshape(-1)
    for i, v in enumerate(t):
        t[i] = f(v)
    return t.reshape(a.shape)

def compute_histogram(image, color_depth):
    '''
    :param image: image to compute histogram. numpy array
    :returns: computed histogram of image. numpy array
    '''
    logging.info('Computing histogram')

    counts = [0 for _ in range(color_depth)]

    for _, i in np.ndenumerate(image):
        counts[i] += 1

    return counts



def histogram_equalize_function(image, color_depth):
    '''
    Computes the histogram equalize function of the input image.
    :param image: the image to equalize
    :param color_depth: the color depth of the image (the number of distinct gray values)
    :returns: an array of length color_depth which represents f(x)
    '''
    logging.info('Equalizing image')

    # get histogram first:
    histogram = compute_histogram(image, color_depth)

    num_pixels = image.shape[0] * image.shape[1]
    normalized_histogram = [i/num_pixels for i in histogram] # ipython 3: returns in floating point for two integers

    func = []

    for i in range(color_depth):
        func.append(math.floor((color_depth-1)*sum(normalized_histogram[:i+1])))

    return func


def main():
    # load image. scipy includes standard lena image
    input_image_depth = 256
    input_image = misc.lena()
    misc.imsave('input_image.png', input_image)


    histogram = compute_histogram(input_image, input_image_depth)
    # imsave histogram as an image

    plt.bar(range(input_image_depth), histogram)
    plt.xlabel('color tone')
    plt.ylabel('# pixels')
    plt.title('Histogram of Lena(original)')
    plt.savefig('histogram_original_lena.png')
    plt.clf()


    # Do histogram equalization
    equalize_function = histogram_equalize_function(input_image, input_image_depth)
    plt.plot(range(input_image_depth), equalize_function)
    plt.title('Histogram equalization color translation function')
    plt.savefig('histogram_equalize_function.png')
    plt.clf()


    equalized_image = array_map(input_image, lambda p: equalize_function[p])
    misc.imsave('image_histogram_equalized.png', equalized_image)

    equalized_histogram = compute_histogram(equalized_image, input_image_depth)
    plt.bar(range(input_image_depth), equalized_histogram)
    plt.xlabel('color tone')
    plt.ylabel('# pixels')
    plt.title('Histogram of Lena(equalized histogram)')
    plt.savefig('histogram_equalized_image.png')
    plt.clf()


if __name__ == '__main__':
    main()
