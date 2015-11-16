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

from skimage import io
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

USE_FFT = 1


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

def fourier(img, inverse = False):
    '''
    Too Slow DFT :(  . Should work though..
    :param img:
    :param inverse:
    :returns: The (inverse) fourier transform of the input image
    '''
    out = np.zeros(img.shape, dtype=complex)
    if inverse:
        sign = 1
    else:
        sign = -1

    for v,u in itertools.product(range(img.shape[0]), range(img.shape[1])):
        for y,x in itertools.product(range(img.shape[0]), range(img.shape[1])):
            out[v][u] += complex(img[y][x])*cmath.exp((1j*complex(sign*2*cmath.pi))*complex(v*y/img.shape[0]+u*x/img.shape[1]))

        if not inverse:
            out[v][u] *= 1/(img.shape[0]*img.shape[1])
    return out

if USE_FFT:
    def fftwrapper(img, inverse=False):
        if inverse:
            return np.fft.ifft2(img)
        else:
            return np.fft.fft2(img)

    fourier = fftwrapper


def power_distance(shape, point):
    return (shape[0]/2-point[0])**2+(shape[1]/2-point[1])**2


def ideal_filter(shape,radius):
    '''
    Returns filter(image) with given shape and radius
    :param shape: A tuple holding the shape of the filter like so: (y,x)
    :param radius: The pixel radius of the ideal filter (i.e. where it should be 1)
    :returns: The image/filter
    '''
    filter_frequency_domain = np.zeros(shape)
    for y,x in itertools.product(range(shape[0]), range(shape[1])):
        if power_distance(shape, (y,x)) <= radius**2:
            filter_frequency_domain[y][x] = 1
    return filter_frequency_domain

def highpass_ideal_filter(shape,radius):
    return 1-ideal_filter(shape,radius)

def butterworth_filter(shape, cutoff_frequency, order = 2):
    '''
    Returns butterworth lowpass filter(image) according to given parameters
    TODO: params here
    '''
    filter_frequency_domain = np.zeros(shape)
    for v,u in itertools.product(range(shape[0]), range(shape[1])):
        filter_frequency_domain[v][u] = 1/(1+(power_distance(shape, (v,u))/cutoff_frequency)**(2*order))

    return filter_frequency_domain

def highpass_butterworth_filter(shape, cutoff_frequency, order=2):
    '''
    Returns butterworth lowpass filter(image) according to given parameters
    TODO: params here
    '''
    filter_frequency_domain = np.zeros(shape)
    for v,u in itertools.product(range(shape[0]), range(shape[1])):
        filter_frequency_domain[v][u] = 1/(1+(power_distance(shape, (v,u))/cutoff_frequency)**(2*order))

    return filter_frequency_domain

def gaussian_filter(shape, cutoff_frequency):
    '''
    Returns butterworth lowpass filter(image) according to given parameters
    TODO: params here
    '''
    filter_frequency_domain = np.zeros(shape)
    for v,u in itertools.product(range(shape[0]), range(shape[1])):
        filter_frequency_domain[v][u] = math.exp((-1)*power_distance(shape, (v,u))/(2*(cutoff_frequency**2)))

    return filter_frequency_domain

def highpass_gaussian_filter(shape, cutoff_frequency):
    return 1-gaussian_filter(shape, cutoff)



def center_scale_image(img):
    '''
    Returns a new image multiplied by -1**(x+y)
    '''
    tmp = img.copy()
    for y,x in itertools.product(range(tmp.shape[0]), range(tmp.shape[1])):
        tmp[y][x] *= (-1)**(x+y)
    return tmp

def scale_to_one(img, color_depth):
    '''
    Scale all values in image between -1 and 1. (At least if all values are in range [-256,256])
    :param img: Input image
    :param color_depth: Number of colors
    '''
    return img / color_depth

def scale_from_one(img, color_depth):
    '''
    Scale all values in image between to [0,256]
    :param img: Input image
    :param color_depth: Number of colors
    '''
    tmp = img.copy()
    tmp = np.abs(tmp) # TODO is this good??
    return tmp


    return tmp * color_depth/np.amax(tmp)

def transform_and_save(filter_frequency_domain, img_frequency_domain, name):
    '''
    Filter the image and bring it back to spatial, then save
    '''
    logging.info('Apply and save filter {}'.format(name))

    # apply filter
    filtered_frequency_domain = np.multiply(img_frequency_domain, filter_frequency_domain)

    # inverse fourier
    filtered_spatial_domain = fourier(filtered_frequency_domain, True)

    # post processing
    filtered_spatial_domain_post_processed = center_scale_image(filtered_spatial_domain).real

    misc.imsave('fourier_filtered_{}.tif'.format(name), filtered_frequency_domain.real, mode=(False, False, True))

    misc.imsave('filter_{}.tif'.format(name), filter_frequency_domain, mode=(True, False, True))

    misc.imsave('result_{}.tif'.format(name), filtered_spatial_domain_post_processed, mode=(True, False, True))


def main():
    # load image. scipy includes standard lena image
    input_image_depth = 256
    input_image = misc.imread('characters_test_pattern.tif')
    shape = input_image.shape

    # im = input_image

    # pre computing
    im_scaled_centered = center_scale_image(input_image)
    # apply fourier
    img_frequency_domain = fourier(im_scaled_centered)

    misc.imsave('fourier.tif', img_frequency_domain.real)

    # Lowpass filters
    for radius in (5,15,30,80,230):
        for filter_function, name in ((ideal_filter, 'lowpass_ideal_{}'), (butterworth_filter, 'lowpass_butterworth_{}'), (gaussian_filter, 'lowpass_gaussian_{}')):
            ideal_filter_frequency_domain = filter_function(shape, radius)
            transform_and_save(ideal_filter_frequency_domain, img_frequency_domain, name.format(radius))


    # Highpass filters
    for radius in (15, 30, 80):
        for filter_function, name in ((highpass_ideal_filter, 'highpass_ideal_{}'), (highpass_butterworth_filter, 'highpass_butterworth_{}'), (highpass_gaussian_filter, 'highpass_gaussian_{}')):
            ideal_filter_frequency_domain = filter_function(shape, radius)

            transform_and_save(ideal_filter_frequency_domain, img_frequency_domain, name.format(radius))

if __name__ == '__main__':
    main()
