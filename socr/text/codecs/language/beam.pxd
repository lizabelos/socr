from .language_model cimport LanguageModel


cdef class Optical:

    cdef float prBlank
    cdef float prNonBlank


cdef class Textual:
    cdef str text
    cdef list wordHist
    cdef str wordDev
    cdef float prUnnormalized
    cdef float prTotal

    cdef copy(self)

cdef class Beam:

    cdef Optical optical
    cdef Textual textual
    cdef LanguageModel lm
    cdef bint useNGrams
    cdef bint newWord

    cpdef mergeBeam(self, Beam beam)

    cdef getText(self)

    cdef getPrBlank(self)

    cdef getPrNonBlank(self)

    cdef getPrTotal(self)

    cdef getPrTextual(self)

    cdef getNextChars(self)

    cdef createChildBeam(self, str newChar, float prBlank, float prNonBlank)

    cdef processNGram(self)


cdef class BeamList:

    cdef dict beams

    cdef addBeam(self, Beam beam)

    cdef sortedLambda(self, Beam x, int lmWeight)

    cdef getBestBeams(self, int num)

    cdef deletePartialBeams(self, LanguageModel lm)

    cdef completeBeams(self, LanguageModel lm)

    cdef dump(self)