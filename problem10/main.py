#!/usr/bin/env python3

import numpy as np
from scipy.linalg import svd
from matplotlib.image import imread, imsave
import matplotlib.pyplot as plt
from matplotlib import cm
import copy


__author__ = 'Moritz Schäfer'
__email__ = 'mail@moritzs.de'
__sjtu_student_id = '713030990015'

fig = plt.figure(figsize=(8, 11), facecolor='white')

class Coord:
    def __init__(self, y, x=None):
        try:
            self.y = y[0]
            self.x = y[1]
        except:
            self.y = y
            self.x = x

    def __repr__(self):
        return 'Coord: <{},{}>'.format(self.y, self.x)

    def __add__(self, other):
        return Coord(self.y+other.y, self.x+other.x)

    def __sub__(self, other):
        return Coord(self.y-other.y, self.x-other.x)
    def __mul__(self, a):
        return Coord(self.y*a, self.x*a)

    def __eq__(self, other):
        try:
            return self.y == other.y and self.x == other.x
        except AttributeError:
            return False

    def __truediv__(self, other):
        return Coord(self.y/other, self.x/other)


    def __tuple__(self):
        return (self.y, self.x)

    def __getitem__(self, i):
        if i == 0:
            return self.y
        elif i == 1:
            return self.x
        else:
            raise IndexError

directions = (
    Coord(0, -1),
    Coord(-1, -1),
    Coord(-1, 0),
    Coord(-1, 1),
    Coord(0, 1),
    Coord(1, 1),
    Coord(1, 0),
    Coord(1, -1),
)

code_directions = (
    Coord(0, 1),
    #Coord(1, 1),
    Coord(-1, 0),
    #Coord(1, -1),
    Coord(0, -1),
    #Coord(-1, -1),
    Coord(1, 0),
    #Coord(-1, 1),
)


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

    return ret

def find_index(l, value):
    for i, v in enumerate(l):
        if value == v:
            return i
    return None
def boundary_following(input_image):
    """ Follows path around image. returns new image with path following as line


    """

    i = np.pad(input_image, 1, mode='constant', constant_values=0)

    points = []
    # search from top left for the left uppermost white pixel
    b = None
    for y in range(i.shape[0]):
        for x in range(i.shape[1]):
            if i[y,x] == 255:
                b = Coord(y, x)
                c = Coord(y, x-1)
                break
        if b is not None:
            break
    if not b:
        raise ValueError('image contains only zeroes')

    def end_of_loop(points, b):
        try:
            return (points[-1] == points[0] and b == points[1])
        except IndexError:
            return False
    while not end_of_loop(points, b):
        cur_dir = directions.index(c-b)

        for _ in range(9):
            if i[tuple(b+directions[cur_dir])] == 255:
                points.append(b)
                b = b+directions[cur_dir]
                c = b+directions[(cur_dir-1)%8]
                break
            else:
                cur_dir = (cur_dir + 1)%8
    return points[:-1]

def image_from_points(shape, points, point_size=1):
    out = np.zeros(shape=shape)
    #out = i.copy()[1:-1,1:-1]

    for p in points:
        for s in range(point_size):
            for d in directions:
                try:
                    out[tuple(p+(d*s))] = 255
                except IndexError:
                    print('why')

    return out

def resample_chain(chain, grid_size=40):

    def resample(p):

        if p.y%grid_size > grid_size/2:
            p.y += (grid_size - (p.y%grid_size))
        else:
            p.y -= p.y%grid_size

        if p.x%grid_size > grid_size/2:
            p.x += (grid_size - (p.x%grid_size))
        else:
            p.x -= p.x%grid_size
        return p
    resampled = [resample(p) for p in copy.deepcopy(chain)]

    densed = []
    last = None
    for p in resampled:
        if p != last:
            densed.append(p)
            last = p

    if densed[-1] == densed[0]:
        return densed[:-1]
    else:
        return densed

def chaincode(points, grid_size = 40):
    code = []
    grid_points = [point/grid_size for point in points]
    for i, p in enumerate(grid_points):
        d = grid_points[(i+1)%len(grid_points)]-p
        if abs(d.y) > 1 or abs(d.x) > 1:
            raise ValueError
        try:
            code.append(code_directions.index(d)+1)
        except ValueError:
            import ipdb; ipdb.set_trace()

    code.insert(0, code[-1])

    return code[:-1]

def diffcode(code, circular=True):
    if circular:
        diff = [(code[0] - code[-1])%4]
    for i, p in enumerate(code[:-1]):
        diff.append((code[i+1]-p)%4)

    return diff


def main():
    # 10 a)
    input_image = custom_imread('noisy_stroke.tif')
    points = boundary_following(input_image)
    draw_img(image_from_points(input_image.shape, points, 1),
             (1, 2),
             'boundary_following.png',
             1)
    resampled = resample_chain(points, 40)
    code = chaincode(resampled, 40)
    diff = diffcode(code)
    print('Unnormalized chaincode: {}'.format(''.join(map(str, code))))
    print('Unnormalized circular first difference: {}'.
          format(''.join(map(str, diff))))

    draw_img(image_from_points(input_image.shape, resampled, 5),
             (1, 2),
             'resampled.png',
             2)
    plt.show()



    # 10 b)
    image_count = 6

    input_image = custom_imread('WashingtonDC_Band1.tif')
    f_matrix = np.ndarray(shape=(input_image.shape[0] * input_image.shape[1],
                          image_count))
    for i in range(image_count):
        f_matrix[:, i] = custom_imread('WashingtonDC_Band{}.tif'.format(i+1)). \
            reshape(-1)

    # calculate mean and subtract from images (center the images)
    input_mean = np.mean(f_matrix, 1)
    for i in range(image_count):
        f_matrix[:, i] -= input_mean

    U, s, Vt = svd(f_matrix, full_matrices=False)
    V = Vt.T

    # sort by singular values so we get the most important PCs
    ind = np.argsort(s)[::-1]
    U = U[:, ind]
    s = s[ind]
    V = V[:, ind]

    # now U contains eigenvectors of f_matrix * f_matrix.T
    # we can now transfer the images

    U_reduced = U[:, :2]
    for i in range(image_count):
        transformed = np.dot(U_reduced.T, f_matrix[:, i])
        back_transformed = np.dot(U_reduced, transformed) + input_mean
        custom_imsave('transformed_{}.png'.format(i+1),
                      back_transformed.reshape(input_image.shape))

        custom_imsave('eigenvec_{}.png'.format(i+1),
                      U[:, i].reshape(input_image.shape))

if __name__ == '__main__':
    main()
