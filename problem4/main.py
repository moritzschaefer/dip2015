#!/usr/bin/env python3

__author__ = 'Moritz Schäfer'
__email__ = 'mail@moritzs.de'
__sjtu_student_id = '713030990015'

'''
Requires python packages: matplotlib, scipy, numpy, pillow, cairocffi
'''


import math
import logging
import itertools

from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

def apply_mask(image, filter_mask_func, mask_shape, scale_to = -1):
    '''
    Applies a filter mask to an image
    :param image: The source image to apply the filter on
    :param filter_mask_func: The filter_mask to apply on the image.
    :param mask_shape: The size of the filter window
    :param scale_to: If greater than zero, scale the output image to contain values only in [0,scale_to]
    :returns: The transformed image
    '''

    # See assignment two
    a = int((mask_shape[0]-1)/2)
    b = int((mask_shape[1]-1)/2)
    filter_indices = list(itertools.product(range(-a,a+1), range(-b,b+1)))


    img_out = np.zeros(image.shape)
    for (y,x), v in np.ndenumerate(image):
        # if x == image.shape[0] or y == image.shape[1]:
        #     continue
        pix_out = 0

        window = np.zeros(mask_shape)

        for s,t in filter_indices:
            if y+s > image.shape[0]-1 or y+s < 0 \
                    or x+t > image.shape[1]-1 or x+t < 0:
                window[s][t] = None
            else:
                window[s][t] = image[y+s, x+t]

        img_out[y,x] = filter_mask_func(window)


    # scale image to be in range [0,256]
    if scale_to > 0:
        # some scaling..
        img_out[img_out<0] = 0
        img_out /= (np.amax(img_out)/scale_to)

    return img_out.astype(int)




def main():
    # load image. scipy includes standard lena image
    input_image_depth = 256
    circuit = misc.imread('Circuit.tif')
    misc.imsave('Circuit.png', circuit)
    shape = circuit.shape



    # TODO: for a full list of noises refer to the slides!!
    # generate noise just by getting "a lot of"(shape) gaussian/uniform distributed random variables
    gaussian_noise = np.random.normal(input_image_depth/2, input_image_depth/3, shape)  # first parameter is mean, second is standard deviation and third parameter is number of values to return # TODO why 2?
    uniform_noise = np.random.normal(0, input_image_depth, shape) # first param

    rayleigh_mode = np.sqrt(2 / np.pi) * input_image_depth/2
    rayleigh_noise = np.random.rayleigh(rayleigh_mode, shape)

    gamma_noise = np.random.gamma(2, input_image_depth/4, shape)

    exponential_noise = np.random.exponential(input_image_depth/2, shape)


    impulse_noise = np.random.rand(shape[0], shape[1])
    impulse_noise[impulse_noise < 0.1] = input_image_depth-1
    impulse_noise[impulse_noise < 0.2] = 0
    impulse_noise[np.logical_and(impulse_noise >= 0.3, impulse_noise <= 1)] = input_image_depth/2

    noises = {'gaussian': gaussian_noise,
              'uniform' : uniform_noise,
              'rayleigh': rayleigh_noise,
              'exponential': exponential_noise,
              'impulse': impulse_noise,
              'gamma': gamma_noise}

    noised_images = {}
    for noise_name, noise_image in noises.items():
        misc.imsave('{}_noise.png'.format(noise_name), noise_image)

        noised = circuit + (noise_image-(input_image_depth/2))/2  # move the mean of the gaussian noise to 0 and half it to reduce its impact on the image
        noised[noised >= input_image_depth] = input_image_depth-1
        noised[noised < 0] = 0
        misc.imsave('{}_circuit.png'.format(noise_name), noised)
        noised_images[noise_name] = noised

    # mean filters

    def arithmetic_mean_filter(window):
        multiplier = 1/(window.shape[0]*window.shape[1])


        return sum([multiplier*v if v and not np.isnan(v) else 0 for _, v in np.ndenumerate(window)])

    def geometric_mean_filter(window):
        mul = 1
        for _, v in np.ndenumerate(window):
            if v and not np.isnan(v):
                mul *= v
        return mul ** (1/(window.shape[0]*window.shape[1]))

    def harmonic_filter(window):
        s = 0
        for _, v in np.ndenumerate(window):
            if v and v != 0 and not np.isnan(v):
                s += 1/v


        try:
            return window.shape[0]*window.shape[1]/s
        except ZeroDivisionError:
            return 0

    def contraharmonic_filter(Q, window):
        '''
        Pass this with an alias like:
            alias window: contraharmonic_filter(3, window)
        '''
        try:
            return sum([v**(Q+1) if v and not np.isnan(v) else 0 for _, v in np.ndenumerate(window)])/sum([v**Q if v and not np.isnan(v) else 0 for _, v in np.ndenumerate(window)])
        except ZeroDivisionError:
            return 0

    def median_filter(window):
        return np.nanmedian(window)
        # same as reshaping window to a list, sort the list and taking the element at len(list)/2

    def max_filter(window):
        return np.nanmax(window)

    def min_filter(window):
        return np.nanmin(window)

    def midpoint_filter(window):
        return (np.nanmax(window)+np.nanmin(window))/2

    def alpha_trimmed_mean_filter(d, window):
        '''
        call with an alias again
        '''
        flat_sorted_window = np.sort(window.reshape(-1))
        flat_sorted_window = flat_sorted_window[np.logical_not(np.isnan(flat_sorted_window))]
        return sum(flat_sorted_window[(d//2):-(d//2)])/(window.shape[0]*window.shape[1] - d)

    filters = {'arithmetic_mean': arithmetic_mean_filter,
               'geometric_mean': geometric_mean_filter,
               'harmonic': harmonic_filter,
               'contraharmonic_1.5': lambda window: contraharmonic_filter(1.5, window),
               'contraharmonic_-1.5': lambda window: contraharmonic_filter(-1.5, window),
               'median': median_filter,
               'max': max_filter,
               'min': min_filter,
               'midpoint': midpoint_filter,
               'alpha_trimmed_mean_5': lambda window: alpha_trimmed_mean_filter(5, window)}

    for filter_size in [5,3]:
        for noise_name in noises:
            for filter_name, filter_func in filters.items():
                filtered_image = apply_mask(noised_images[noise_name], filter_func, (filter_size,filter_size))
                try:
                    misc.imsave('noise_{}_filter_{}_size_{}.png'.format(noise_name, filter_name, filter_size), filtered_image)
                except Exception as e:
                    import ipdb; ipdb.set_trace()
                    logging.error('Could not save noise_{}_filter_{}.png: {}'.format(noise_name, filter_name, e))

if __name__ == '__main__':
    main()
