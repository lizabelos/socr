from __future__ import division
from __future__ import print_function

from .beam cimport Beam, BeamList
from .language_model cimport LanguageModel


cpdef wordBeamSearch(float[:,:] mat, int beamWidth, LanguageModel lm, bint useNGrams):
    "decode matrix, use given beam width and language model"
    cdef str chars = lm.getAllChars()
    cdef int blankIdx = 0
    # blankIdx = len(chars)  # blank label is supposed to be last label in RNN output
    cdef int maxT = mat.shape[0] # shape of RNN output: TxC

    cdef Beam genesisBeam = Beam(lm, useNGrams)  # empty string
    cdef BeamList last = BeamList()  # list of beams at time-step before beginning of RNN output
    last.addBeam(genesisBeam)  # start with genesis beam

    cdef BeamList curr
    cdef Beam beam
    cdef list bestBeams

    cdef float prNonBlank
    cdef float prBlank
    cdef int labelIdx

    cdef str nextChars
    cdef str c

    # go over all time-steps
    for t in range(maxT):
        curr = BeamList()  # list of beams at current time-step

        # go over best beams
        bestBeams = last.getBestBeams(beamWidth)  # get best beams

        for beam in bestBeams:
            # calc probability that beam ends with non-blank
            prNonBlank = 0
            if beam.getText() != '':
                # char at time-step t must also occur at t-1
                labelIdx = chars.index(beam.getText()[-1])
                prNonBlank = beam.getPrNonBlank() * mat[t, labelIdx]

            # calc probability that beam ends with blank
            prBlank = beam.getPrTotal() * mat[t, blankIdx]

            # save result
            curr.addBeam(beam.createChildBeam('', prBlank, prNonBlank))

            # extend current beam with characters according to language model
            nextChars = beam.getNextChars()

            for c in nextChars:
                # extend current beam with new character
                labelIdx = chars.index(c)
                if beam.getText() != '' and beam.getText()[-1] == c:
                    prNonBlank = mat[t, labelIdx] * beam.getPrBlank()  # same chars must be separated by blank
                else:
                    prNonBlank = mat[t, labelIdx] * beam.getPrTotal()  # different chars can be neighbours

                # save result
                curr.addBeam(beam.createChildBeam(c, 0, prNonBlank))

        # move current beams to next time-step
        last = curr

    # return most probable beam
    last.completeBeams(lm)
    bestBeams = last.getBestBeams(1)  # sort by probability
    cdef Beam bestBeam = bestBeams[0]
    return bestBeam.getText()