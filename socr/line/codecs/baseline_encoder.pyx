import numpy as np


cdef class BaselineEncoder:

    cdef float height_factor
    cdef int thicknesses

    def __init__(self, float height_factor, int thicknesses):
        self.height_factor = height_factor
        self.thicknesses = thicknesses

    cdef plot(self, float[:,:,:] y_true, int x, int y, int height):
        if x >= y_true.shape[1] or y >= y_true.shape[0] or x < 0 or y < 0:
            return

        y_true[y][x][0] = 1.0
        y_true[y][x][1] = height * self.height_factor

    cdef plot_radius(self, float[:,:,:] y_true, int x, int y, int height):
        cdef int radius = self.thicknesses
        if radius == 0:
            radius = height // 2
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                self.plot(y_true, x + i, y + j, height)

    cdef draw_vertical_line(self, float[:,:,:] y_true, int[:] line):
        cdef int x0 = line[0]
        cdef int y0 = line[1]
        cdef int x1 = line[2]
        cdef int y1 = line[3]
        cdef int height = line[4]

        for y in range(y0, y1):
            self.plot_radius(y_true, x0, y, height)

    cdef draw_line(self, float[:,:,:] y_true, int[:] line):
        cdef int x0 = line[0]
        cdef int y0 = line[1]
        cdef int x1 = line[2]
        cdef int y1 = line[3]
        cdef int height = line[4]

        cdef int deltax = x1 - x0
        cdef int deltay = y1 - y0
        if deltax == 0:
            return self.draw_vertical_line(y_true, line)
        cdef float deltaerr = abs(float(deltay) / float(deltax))
        cdef float error = 0.0
        cdef int y = y0

        for x in range(x0, x1):
            self.plot_radius(y_true, x, y, height)
            error = error + deltaerr
            while error >= 0.5:
                sign = -1 if deltay < 0 else 1
                y = y + sign
                error = error - 1.0

    cpdef encode(self, int[:] image_size, int[:,:] base_lines):
        cdef float[:,:,:] y_true = np.zeros((image_size[1], image_size[0], 2), dtype=np.float32)

        for i in range(0, base_lines.shape[0]):
            self.draw_line(y_true, base_lines[i])

        return np.asarray(y_true)