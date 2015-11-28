#!/usr/bin/env python3

import math
import numpy as np
from scipy.signal import convolve2d
import scipy

from PIL import Image

from matplotlib.image import imread, imsave
import matplotlib.pyplot as plt
from matplotlib import cm

__author__ = 'Moritz Schäfer'
__email__ = 'mail@moritzs.de'
__sjtu_student_id = '713030990015'

fig = plt.figure(figsize=(8, 11), facecolor='white')


def draw_img(image, dim, title, pos, save=True):
    ax = fig.add_subplot(dim[0], dim[1], pos)
    ax.imshow(image, cmap=cm.Greys_r)
    ax.axis('off')
    ax.set_title(title)
    if save:
        custom_imsave(title, image)


def custom_imread(path):
    return imread(path)


def custom_imsave(path, image):
    imsave(path, image, cmap=cm.Greys_r)
    return

    ret = image.copy()
    ret = ret / (ret.max()/255.0)
    imsave(path, ret.astype(int))


def prewitt(A):
    kernel_y = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1],
    ])
    kernel_x = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1],
    ])
    g_y = convolve2d(A, kernel_y)  # already implemented in earlier assignments
    g_x = convolve2d(A, kernel_x)
    magnitude = np.sqrt(np.multiply(g_x**2, g_y**2))

    # apply logarithmic scale
    return 1.4**np.log2(magnitude)


def sobel(A):
    # almost same as prewitt
    kernel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1],
    ])
    kernel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ])
    g_y = convolve2d(A, kernel_y)  # already implemented in earlier assignments
    g_x = convolve2d(A, kernel_x)
    magnitude = np.sqrt(np.multiply(g_x**2, g_y**2))

    return 1.4**np.log2(magnitude)


def roberts(A):
    # almost same as prewitt and sobel
    # almost same as prewitt
    kernel_y = np.array([
        [0, 1],
        [-1, 0],
    ])
    kernel_x = np.array([
        [1, 0],
        [0, -1],
    ])
    g_y = convolve2d(A, kernel_y)  # already implemented in earlier assignments
    g_x = convolve2d(A, kernel_x)
    magnitude = np.sqrt(np.multiply(g_x**2, g_y**2))

    return 1.4**np.log2(magnitude)


def marrhildreth(A):

    gaussian_filter = np.array([
        [0, 0, 1, 0, 0],
        [0, 1, 2, 1, 0],
        [1, 2, -16, 2, 1],
        [0, 1, 2, 1, 0],
        [0, 0, 1, 0, 0]])

    convoluted = convolve2d(A, gaussian_filter)

    height, width = convoluted.shape

    tmp = np.zeros(convoluted.shape)

    for i in range(1, height-1):
        for j in range(1, width-1):
            if convoluted[i, j]:
                if (convoluted[i, j+1] >= 0 and convoluted[i, j-1] < 0) or \
                        (convoluted[i, j+1] < 0 and convoluted[i, j-1] >= 0):
                    tmp[i, j] = convoluted[i, j+1]
                elif (convoluted[i+1, j] >= 0 and convoluted[i-1, j] < 0) or \
                        (convoluted[i+1, j] < 0 and convoluted[i-1, j] >= 0):
                    tmp[i, j] = convoluted[i, j+1]
                elif (convoluted[i+1, j+1] >= 0 and convoluted[i-1, j-1] < 0) \
                        or (convoluted[i+1, j+1] < 0 and
                            convoluted[i-1, j-1] >= 0):
                    tmp[i, j] = convoluted[i, j+1]
                elif (convoluted[i-1, j+1] >= 0 and convoluted[i+1, j-1] < 0) \
                        or (convoluted[i-1, j+1] < 0 and
                            convoluted[i+1, j-1] >= 0):
                    tmp[i, j] = convoluted[i, j+1]

    return tmp


def canny(input_image):
    # convenience functions
    def nonmaxs(grad, i, j, x1, y1, x2, y2):
        try:
            if (grad[i, j] > grad[i+x1, j+y1]
                    ) and (grad[i, j] > grad[i+x2, j+y2]):
                return 1
            else:
                return 0
        except IndexError:
            return -1

    def edge_end(imag, threshold):
        X, Y = np.where(imag > threshold)
        try:
            y = Y.min()
        except:
            return -1
        X = X.tolist()
        Y = Y.tolist()
        index = Y.index(y)
        x = X[index]
        return [x, y]

    def edge_follower(imag, p0, p1, p2, threshold):
        kit = [-1, 0, 1]
        X, Y = imag.shape
        for i in kit:
            for j in kit:
                if (i+j) == 0:
                    continue
                x = p0[0]+i
                y = p0[1]+j

                if (x < 0) or (y < 0) or (x >= X) or (y >= Y):
                    continue
                if ([x, y] == p1) or ([x, y] == p2):
                    continue
                if (imag[x, y] > threshold):  # and (imag[i,j] < 256):
                    return [x, y]
        return -1
    thresholdhold_high = 40
    thresholdhold_low = 8

    # precalculated gauss kernel (sigma=0.5, window size 5)

    gausskernel = np.array([
        [  6.96247819e-08,   2.80886418e-05,   2.07548550e-04,   2.80886418e-05,
            6.96247819e-08],
        [  2.80886418e-05,   1.13317669e-02,   8.37310610e-02,   1.13317669e-02,
            2.80886418e-05],
        [  2.07548550e-04,   8.37310610e-02,   6.18693507e-01,   8.37310610e-02,
            2.07548550e-04],
        [  2.80886418e-05,   1.13317669e-02,   8.37310610e-02,   1.13317669e-02,
            2.80886418e-05],
        [  6.96247819e-08,   2.80886418e-05,   2.07548550e-04,   2.80886418e-05,
            6.96247819e-08]
    ])

    fx = np.array([
        [1, 1, 1,],
        [0, 0, 0,],
        [-1, -1, -1]])
    fy = np.array([
        [-1, 0, 1,],
        [-1, 0, 1,],
        [-1, 0, 1]])

    imout = convolve2d(input_image, gausskernel)[1:-1, 1:-1]
    gradx = convolve2d(imout, fx)[1:-1, 1:-1]
    grady = convolve2d(imout, fy)[1:-1, 1:-1]


    grad = scipy.hypot(gradx, grady)
    theta = scipy.arctan2(grady, gradx)
    theta = 180 + (180/math.pi)*theta
    x, y = np.where(grad < 10)
    theta[x, y] = 0
    grad[x, y] = 0

    x0, y0 = np.where(((theta < 22.5)+(theta > 157.5)*(theta < 202.5) +
                        (theta > 337.5)) == True)
    x45, y45 = np.where(((theta > 22.5)*(theta < 67.5) +
                        (theta > 202.5)*(theta < 247.5)) == True)
    x90, y90 = np.where(((theta > 67.5)*(theta < 112.5) +
                        (theta > 247.5)*(theta < 292.5)) == True)
    x135, y135 = np.where(((theta > 112.5)*(theta < 157.5) +
                        (theta > 292.5)*(theta < 337.5)) == True)

    Image.fromarray(theta).convert('L').save('Angle map.jpg')
    theta[x0, y0] = 0
    theta[x45, y45] = 45
    theta[x90, y90] = 90
    theta[x135, y135] = 135
    x, y = theta.shape
    temp = Image.new('RGB', (y, x), (255, 255, 255))
    for i in range(x):
        for j in range(y):
            if theta[i, j] == 0:
                temp.putpixel((j, i), (0, 0, 255))
            elif theta[i, j] == 45:
                temp.putpixel((j, i), (255, 0, 0))
            elif theta[i, j] == 90:
                temp.putpixel((j, i), (255, 255, 0))
            elif theta[i, j] == 45:
                temp.putpixel((j, i), (0, 255, 0))
    grad = grad.copy()
    x, y = grad.shape

    for i in range(x):
        for j in range(y):
            if theta[i, j] == 0:
                if not nonmaxs(grad, i, j, 1, 0, -1, 0):
                    grad[i, j] = 0

            elif theta[i, j] == 45:
                if not nonmaxs(grad, i, j, 1, -1, -1, 1):
                    grad[i, j] = 0

            elif theta[i, j] == 90:
                if not nonmaxs(grad, i, j, 0, 1, 0, -1):
                    grad[i, j] = 0
            elif theta[i, j] == 135:
                if not nonmaxs(grad, i, j, 1, 1, -1, -1):
                    grad[i, j] = 0

    init_point = edge_end(grad, thresholdhold_high)

    while (init_point != -1):
        grad[init_point[0], init_point[1]] = -1
        p2 = init_point
        p1 = init_point
        p0 = init_point
        p0 = edge_follower(grad, p0, p1, p2, thresholdhold_low)

        while (p0 != -1):
            p2 = p1
            p1 = p0
            grad[p0[0], p0[1]] = -1
            p0 = edge_follower(grad, p0, p1, p2, thresholdhold_low)

        init_point = edge_end(grad, thresholdhold_high)

    x, y = np.where(grad == -1)
    grad[:, :] = 0
    grad[x, y] = 255
    return grad


def otsu(A):
    n = 256
    num_pixels = A.shape[0] * A.shape[1]
    hist = np.histogram[A, n]
    s = 0
    weight_b = 0
    maximum = 0
    thresholdholds = [0, 0]
    t_sum = sum(np.multiply(np.array(range(n)), hist.T))
    for i in range(n):
        weight_b = weight_b + hist[i]
        if weight_b == 0:
            continue
        if num_pixels - weight_b == 0:
            break
        s = s + i * hist[i]
        tmp = (t_sum - s) / (num_pixels - weight_b)
        between = weight_b * (num_pixels - weight_b) * \
            ((s / weight_b) - tmp) * ((s / weight_b) - tmp)
        if between >= maximum:
            thresholdholds[0] = i
            if between > maximum:
                thresholdholds[1] = i
            maximum = between
    thresholdhold = sum(thresholdholds)/(2)
    tmp = A.copy()
    tmp[tmp < thresholdhold] = 0
    tmp[tmp >= thresholdhold] = 1

    return tmp


def main():
    # 9 a)
    input_image = custom_imread('building.tif')
    input_image2 = custom_imread('polymersomes.tif')

    draw_img(canny(input_image), (2, 3), 'canny.png', 5)
    draw_img(prewitt(input_image), (2, 3), 'prewitted.jpg', 1)
    draw_img(sobel(input_image), (2, 3), 'sobel.png', 2)
    draw_img(roberts(input_image), (2, 3), 'roberts.png', 3)
    draw_img(marrhildreth(input_image), (2, 3), 'marrhildreth.png', 4)

    # 9 b)
    #draw_img(otsu(input_image2), (2, 3), 'otsu.png', 6)

    plt.show()


if __name__ == '__main__':
    main()
