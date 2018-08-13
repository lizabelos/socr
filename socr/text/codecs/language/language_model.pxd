from .prefix_tree cimport PrefixTree


cdef class LanguageModel:

    cdef str wordCharPattern
    cdef str wordPattern
    cdef int numWords
    cdef int numUniqueWords
    cdef bint smoothing
    cdef float addK
    cdef PrefixTree tree
    cdef str allChars
    cdef str wordChars
    cdef str nonWordChars

    cdef str lastW1
    cdef str lastW2
    cdef object lastProb

    cdef object nnGram

    cdef getNextWords(self, text)

    cdef getNextChars(self, text)

    cdef getWordChars(self)

    cdef getNonWordChars(self)

    cdef getAllChars(self)

    cdef getWordCharPattern(self)

    cdef isWord(self, text)

    cdef getNN(self)

    cdef getBigramProb(self, w1, w2)