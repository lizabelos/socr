#cython: wraparound=False
#cython: boundscheck=False
#cython: cdivision=True

import math

import numpy as np
cimport numpy as np

from socr.utils.image.connected_components import connected_components


cdef class BaselineDecoder:

    cdef float height_factor

    def __init__(self, height_factor):
        self.height_factor = height_factor

    cpdef list decode(self, double[:,:,:] image, float[:,:,:] predicted, bint with_images=True, int degree=3):
        cdef np.ndarray[int, ndim=2] components = connected_components(predicted[0])
        cdef int num_components = np.max(components)
        cdef list results = []

        for i in range(1, num_components + 1):
            result = self.process_components(image, predicted, components, i, with_image=with_images, degree=degree, line_height=64, baseline_resolution=16)
            if result is not None:
                results.append(result)

        return results

    cdef tuple process_components(self, double[:,:,:] image, float[:,:,:] prediction, int[:,:] components, int index, int degree=3, int line_height=64, int baseline_resolution=16, bint with_image=True):
        cdef list x_train = []
        cdef list y_data = []

        cdef int max_height = 10

        for y in range(0, components.shape[0]):
            for x in range(0, components.shape[1]):
                if components[y][x] == index:
                    x_train.append(x)
                    y_data.append(y)
                    max_height = int(max(max_height, prediction[1][y][x] / self.height_factor))

        if len(x_train) == 0:
            return None

        max_height = max_height * 2

        cdef float min_x = min(x_train)
        cdef float max_x = max(x_train)
        cdef float line_components_width = max_x - min_x
        cdef int line_components_height = int(max_height)
        cdef int line_width = int(line_components_width * line_height / line_components_height)

        if line_components_width < 10 and line_components_height < 10:
            # print("Warning : too small image width and height for index : " + str(index) + ". Skipping.")
            return None

        if line_components_width < 10:
            # print("Warning : too small image width for index : " + str(index) + ". Skipping.")
            return None

        if line_components_height < 10:
            # print("Warning : too small image height for index : " + str(index) + ". Skipping.")
            return None

        cdef double[:] coefs = np.polynomial.polynomial.polyfit(np.array(x_train), np.array(y_data), degree)
        ffit = np.polynomial.Polynomial(coefs)
        fderiv = np.polynomial.Polynomial(np.polynomial.polynomial.polyder(coefs))

        # fnorm = lambda x, a: -(a - x) / fderiv(x) + ffit(x)

        cdef list baseline_points = []

        for i in range(0, baseline_resolution + 1):
            x = min_x * (i / baseline_resolution) + max_x * ((baseline_resolution - i) / baseline_resolution)
            baseline_points.append(int(x * image.shape[2] / prediction.shape[2]))
            baseline_points.append(int(ffit(x) * image.shape[1] / prediction.shape[1]))

        line = None

        if with_image:
            line = np.ones((3, line_height, line_width))
            for line_x in range(0, line_width):
                x = line_x * line_components_width / line_width + min_x

                deriv = fderiv(x)

                if deriv == 0:
                    tanx0 = x
                    tany0 = ffit(x) + 1
                    tanx1 = x
                    tany1 = ffit(x) - 1
                else:
                    tanx0 = x + 1
                    tany0 = -((x + 1) - x) / fderiv(x) + ffit(x)
                    # tany0 = fnorm(x, x + 1)
                    tanx1 = x - 1
                    tany1 = -((x - 1) - x) / fderiv(x) + ffit(x)
                    # tany1 = fnorm(x, x - 1)
                    if tany0 < tany1:
                        tanx2, tany2 = tanx0, tany0
                        tanx0, tany0 = tanx1, tany1
                        tanx1, tany1 = tanx2, tany2

                dist0 = math.sqrt((tanx0 - x) ** 2 + (tany0 - ffit(x)) ** 2)
                dist1 = math.sqrt((tanx1 - x) ** 2 + (tany1 - ffit(x)) ** 2)

                tanx0 = (tanx0 - x) * line_components_height / (dist0 * 2) + x
                tany0 = (tany0 - ffit(x)) * line_components_height / (dist0 * 2) + ffit(x)
                tanx1 = (tanx1 - x) * line_components_height / (dist1 * 2) + x
                tany1 = (tany1 - ffit(x)) * line_components_height / (dist1 * 2) + ffit(x)

                for line_y in range(0, line_height):
                    y = line_y * line_components_height / line_height

                    # image_x = x * image.shape[2] / prediction.shape[2]
                    image_x = tanx0 * (y / line_components_height) + tanx1 * (
                                (line_components_height - y) / line_components_height)
                    image_x = image_x * image.shape[2] / prediction.shape[2]

                    image_y = tany0 * (y / line_components_height) + tany1 * (
                                (line_components_height - y) / line_components_height)
                    image_y = image_y * image.shape[1] / prediction.shape[1]

                    if image_x < 0 or image_y < 0 or image_x >= image.shape[2] or image_y >= image.shape[1]:
                        continue
                    # image_y = y * (tany0 - tany1) / line_height + tany1

                    for c in range(0, 3):
                        line[c][int(line_y)][int(line_x)] = image[c][int(image_y)][int(image_x)]

        return line, baseline_points
