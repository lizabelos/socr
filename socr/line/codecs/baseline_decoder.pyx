import math

import numpy as np

from socr.utils.image import connected_components


cdef class BaselineDecoder:

    cdef float height_factor
    cdef float hysteresis_minimum
    cdef float hysteresis_maximum

    def __init__(self, height_factor, hysteresis_minimum, hysteresis_maximum):
        self.height_factor = height_factor
        self.hysteresis_minimum = hysteresis_minimum
        self.hysteresis_maximum = hysteresis_maximum

    cpdef set_hysteresis(self, hysteresis_minimum, hysteresis_maximum):
        self.hysteresis_minimum = hysteresis_minimum
        self.hysteresis_maximum = hysteresis_maximum

    cpdef tuple decode(self, double[:,:,:] image, float[:,:,:] predicted, bint with_images=True, int degree=3, bint brut_points=False):
        cdef int[:,:] components = connected_components(predicted[0], hist_min=self.hysteresis_minimum, hist_max=self.hysteresis_maximum)
        cdef int num_components = np.max(components)
        cdef list results = []

        for i in range(1, num_components + 1):
            result = self.process_components(image, predicted, components, i, with_image=with_images, degree=degree, line_height=64, baseline_resolution=16, brut_points=brut_points)
            if result is not None:
                results.append(result)

        return results, components

    cdef tuple process_components(self, double[:,:,:] image, float[:,:,:] prediction, int[:,:] components, int index, int degree=3, int line_height=64, int baseline_resolution=16, bint with_image=True, bint brut_points=False):
        if prediction.shape[1] != components.shape[0] or prediction.shape[2] != components.shape[1]:
            print(str(prediction.shape) + "!=" + str(components.shape))
            raise AssertionError


        cdef list x_train = []
        cdef list y_data = []

        cdef int max_height = 10

        cdef double[:] x_maxs = np.zeros((components.shape[1]))
        cdef double[:] x_maxs_pos = np.zeros((components.shape[1]))

        cdef int x
        cdef int y

        for y in range(0, components.shape[0]):
            for x in range(0, components.shape[1]):

                if components[y][x] == index:
                    x_train.append(x)
                    y_data.append(y)
                    max_height = int(max(max_height, prediction[1][y][x] / self.height_factor))

                    x_maxs[x] = x_maxs[x] + (1 * y)
                    x_maxs_pos[x] = x_maxs_pos[x] + 1

        if len(x_train) == 0:
            return None

        max_height = max_height * 2

        cdef int min_x = min(x_train)
        cdef int max_x = max(x_train)
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

        if brut_points:
            for i in range(0, baseline_resolution + 1):
                x = min_x * (i / baseline_resolution) + max_x * ((baseline_resolution - i) / baseline_resolution)
                baseline_points.append(int(x * image.shape[2] / prediction.shape[2]))
                baseline_points.append(int(ffit(x) * image.shape[1] / prediction.shape[1]))
        else:
            for x in range(min_x, max_x):
               if x_maxs_pos[x] != 0:
                   y = int(x_maxs[x] / x_maxs_pos[x])
                   baseline_points.append(x * image.shape[2] / prediction.shape[2])
                   baseline_points.append(y * image.shape[1] / prediction.shape[1])
            # for i in range(9, len(x_train)):
            #     baseline_points.append(int(x_train[i] * image.shape[2] / prediction.shape[2]))
            #     baseline_points.append(int(y_data[i] * image.shape[1] / prediction.shape[1]))

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
