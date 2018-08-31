from socr.utils.logger import print_error


cdef class CTCDecoder:

    cdef dict inv_labels
    cdef int label_len

    def __init__(self, inv_labels, label_len):
        self.inv_labels = inv_labels
        self.label_len = label_len

        # Intialize corpus and language model

    cpdef decode(self, float[:,:,:] sequence):
        # OUTPUT : batch_size x width x num_label
        if sequence.shape[2] != self.label_len:
            print_error(str(sequence.shape) + "!=" + str(self.label_len))
            assert False

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
    
    
    cpdef beam_bi_decode(self, float[:,:,:] sequence):
        cdef int num_ngram = 64

        if sequence.shape[2] != self.label_len:
            print_error(str(sequence.shape) + "!=" + str(self.label_len))
            assert False

        cdef int width = sequence.shape[1]
        cdef int batch_size = sequence.shape[0]

        cdef str text = ""
        cdef int last_label = -1

        cdef int time
        cdef int max_label
        cdef int i

        cdef list last_beams = [(1.0,[])]
        cdef list current_beams

        for time in range(0, width):

            current_beams = []

            for beam in last_beams:
                for i in range(0, self.label_len):
                    # TODO : blank and repetition !
                    # TODO : take care about the restriction
                    current_beams.append((beam[0] * sequence[0][time][i],beam[1] + [self.inv_labels[i]]))


            last_beams = sorted(current_beams, reverse=True, key=lambda x: x[0])[:num_ngram]




            # TODO : think to normalize


        return last_beams[0][1]