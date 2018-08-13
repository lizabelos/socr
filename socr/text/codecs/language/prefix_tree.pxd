cdef class Node:
    cdef dict children
    cdef bint isWord

cdef class PrefixTree:

    cdef Node root

    cdef addWord(self, str text)

    cdef addWords(self, list words)

    cdef getNode(self, str text)

    cdef isWord(self, str text)

    cdef getNextChars(self, str text)

    cdef getNextWords(self, str text)

    cdef dump(self)