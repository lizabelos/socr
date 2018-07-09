#cython: wraparound=False
#cython: boundscheck=False
#cython: cdivision=True

from __future__ import print_function
from __future__ import division


cdef class Node:
    "class representing nodes in a prefix tree"

    def __init__(self):
        self.children = {}  # all child elements beginning with current prefix
        self.isWord = False  # does this prefix represent a word

    def __str__(self):
        s = ''
        for (k, _) in self.children.items():
            s += k
        return 'isWord: ' + str(self.isWord) + '; children: ' + s


cdef class PrefixTree:
    "prefix tree"

    def __init__(self):
        self.root = Node()

    cdef addWord(self, str text):
        "add word to prefix tree"
        cdef Node node = self.root
        for i in range(len(text)):
            c = text[i]  # current char
            if c not in node.children:
                node.children[c] = Node()
            node = node.children[c]
            isLast = (i + 1 == len(text))
            if isLast:
                node.isWord = True

    cdef addWords(self, list words):
        for w in words:
            self.addWord(w)

    cdef getNode(self, str text):
        "get node representing given text"
        cdef Node node = self.root
        for c in text:
            if c in node.children:
                node = node.children[c]
            else:
                return None
        return node

    cdef isWord(self, str text):
        cdef Node node = self.getNode(text)
        if node:
            return node.isWord
        return False

    cdef getNextChars(self, str text):
        "get all characters which may directly follow given text"
        chars = []
        cdef Node node = self.getNode(text)
        if node:
            for k, _ in node.children.items():
                chars.append(k)
        return chars

    cdef getNextWords(self, str text):
        "get all words of which given text is a prefix (including the text itself, it is a word)"
        cdef list words = []
        cdef list prefixes
        cdef list nodes
        cdef Node node = self.getNode(text)
        cdef Node currNode
        if node:
            nodes = [node]
            prefixes = [text]
            while len(nodes) > 0:
                # put all children into list
                currNode = nodes[0]
                for k, v in currNode.children.items():
                    nodes.append(v)
                    prefixes.append(prefixes[0] + k)

                # is current node a word
                if currNode.isWord:
                    words.append(prefixes[0])

                # remove current node
                del nodes[0]
                del prefixes[0]

        return words

    cdef dump(self):
        nodes = [self.root]
        while len(nodes) > 0:
            # put all children into list
            for _, v in nodes[0].children.items():
                nodes.append(v)

            # dump current node
            print(nodes[0])

            # remove from list
            del nodes[0]