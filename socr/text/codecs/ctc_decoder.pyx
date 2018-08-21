cdef class CTCNode:

    cdef str c
    cdef list childs
    cdef str next

    def __init__(self, c):
        self.c = c
        self.childs = []

        self.next = ""
        if len(self.c) > 1:
            self.next = self.c[1:]

    def create_childs(self, sequence, i):
        pass


cdef class CTCDecoder:

    cdef dict inv_labels
    cdef int label_len

    def __init__(self, inv_labels, label_len):
        self.inv_labels = inv_labels
        self.label_len = label_len

    cpdef decode(self, float[:,:,:] sequence):
        # OUTPUT : batch_size x width x num_label
        assert sequence.shape[2] == self.label_len

        cdef int width = sequence.shape[1]
        cdef int batch_size = sequence.shape[0]

        cdef str text = ""
        cdef int last_label = -1

        cdef int time
        cdef int max_label
        cdef int i

        for time in range(0, width):

            max_label = 0
            for i in range(1, self.label_len):
                if sequence[0][time][i] > sequence[0][time][max_label]:
                    max_label = i

            if max_label != last_label:
                text = text + self.inv_labels[max_label]
                last_label = max_label

        return text

    cpdef decode2(self, float[:,:,:] sequence):
        assert sequence.shape[2] == self.label_len

        cdef int width = sequence.shape[1]
        cdef int batch_size = sequence.shape[0]

        return


