#cython: boundscheck=False
#cython: cdivision=True

from __future__ import print_function
from __future__ import division

from .language_model cimport LanguageModel


cdef class Optical:
    "optical score of beam"

    def __init__(self, prBlank=0.0, prNonBlank=0.0):
        self.prBlank = prBlank  # prob of ending with a blank
        self.prNonBlank = prNonBlank  # prob of ending with a non-blank


cdef class Textual:
    "textual score of beam"

    def __init__(self, text=''):
        self.text = text
        self.wordHist = []  # history of words so far
        self.wordDev = ''  # developing word
        self.prUnnormalized = 1.0
        self.prTotal = 1.0

    cdef copy(self):
        cdef Textual textual = Textual()
        textual.text = self.text
        textual.wordHist = self.wordHist.copy()
        textual.wordDev = self.wordDev
        textual.prUnnormalized = self.prUnnormalized
        textual.prTotal = self.prTotal
        return textual


cdef class Beam:
    "beam with text, optical and textual score"

    def __init__(self, lm, useNGrams):
        "creates genesis beam"
        self.optical = Optical(1.0, 0.0)
        self.textual = Textual('')
        self.lm = lm
        self.useNGrams = useNGrams
        self.newWord = False

    cpdef mergeBeam(self, Beam beam):
        "merge probabilities of two beams with same text"

        if self.getText() != beam.getText():
            raise Exception('mergeBeam: texts differ')

        self.optical.prNonBlank += beam.getPrNonBlank()
        self.optical.prBlank += beam.getPrBlank()

    cdef getText(self):
        return self.textual.text

    cdef getPrBlank(self):
        return self.optical.prBlank

    cdef getPrNonBlank(self):
        return self.optical.prNonBlank

    cdef getPrTotal(self):
        return self.getPrBlank() + self.getPrNonBlank()

    cdef getPrTextual(self):
        return self.textual.prTotal

    cdef getNextChars(self):
        return self.lm.getNextChars(self.textual.wordDev)

    cdef createChildBeam(self, str newChar, float prBlank, float prNonBlank):
        "extend beam by new character and set optical score"
        beam = Beam(self.lm, self.useNGrams)

        # copy textual information
        beam.textual = self.textual.copy()
        beam.textual.text += newChar

        # do textual calculations only if beam gets extended
        if newChar != '':

            # if new char occurs inside a word
            if newChar in beam.lm.getWordChars():
                beam.textual.wordDev += newChar
            # if new char does not occur inside a word
            else:
                # if current word is not empty, add it to history
                if beam.textual.wordDev != '':
                    beam.textual.wordHist.append(beam.textual.wordDev)
                    beam.textual.wordDev = ''
                    beam.newWord = True
                    # beam.processNGram()

                if newChar != ' ':
                    beam.textual.wordHist = []



        # set optical information
        beam.optical.prBlank = prBlank
        beam.optical.prNonBlank = prNonBlank
        return beam

    cdef processNGram(self):
        # score with n-grams with n > 1
        return
        if self.newWord:
            numWords = len(self.textual.wordHist)
            if numWords >= 2:
                # todo : test the last word in dictionnary
                # todo : then calculate n gram
                bigramProb = self.lm.getNN().get_ngram_prob(self.textual.wordHist[0:len(self.textual.wordHist)-1], self.textual.wordHist[-1])
                if bigramProb is not None:
                    self.textual.prUnnormalized *= bigramProb
                    self.textual.prTotal = self.textual.prUnnormalized ** (1 / numWords)

    def __str__(self):
        return '"' + self.getText() + '"' + ';' + str(self.getPrTotal()) + ';' + str(self.getPrTextual()) + ';' + str(self.textual.prUnnormalized)


cdef class BeamList:
    "list of beams at specific time-step"

    def __init__(self):
        self.beams = {}

    cdef addBeam(self, Beam beam):
        "add or merge new beam into list"
        # add if text not yet known
        if beam.getText() not in self.beams:
            self.beams[beam.getText()] = beam
        # otherwise merge with existing beam
        else:
            self.beams[beam.getText()].mergeBeam(beam)

    cdef sortedLambda(self, Beam x, int lmWeight):
        return x.getPrTotal() * (x.getPrTextual() ** lmWeight)

    cdef getBestBeams(self, int num):
        "return best beams, specify the max. number of beams to be returned (beam width)"
        cdef int numNGram = num * 2
        u = [v for (_, v) in self.beams.items()]
        lmWeight = 1
        u = sorted(u, reverse=True, key=lambda x: self.sortedLambda(x, lmWeight))[:numNGram]

        cdef Beam beam
        for beam in u:
            beam.processNGram()

        u = sorted(u, reverse=True, key=lambda x: self.sortedLambda(x, lmWeight))[:num]
        return u

    cdef deletePartialBeams(self, LanguageModel lm):
        "delete beams for which last word is not finished"
        for (k, v) in self.beams.items():
            lastWord = v.textual.wordDev
            if (lastWord != '') and (not lm.isWord(lastWord)):
                del self.beams[k]

    cdef completeBeams(self, LanguageModel lm):
        "complete beams such that last word is complete word"
        cdef Beam v
        for (_, v) in self.beams.items():
            lastPrefix = v.textual.wordDev
            if lastPrefix == '' or lm.isWord(lastPrefix):
                continue

            # get word candidates for this prefix
            words = lm.getNextWords(lastPrefix)
            # if there is just one candidate, then the last prefix can be extended to
            if len(words) == 1:
                word = words[0]
                v.textual.text += word[len(lastPrefix) - len(word):]

    cdef dump(self):
        for k in self.beams.keys():
            print(unicode(self.beams[k]).encode('ascii', 'replace'))  # map to ascii if possible (for py2 and windows)
