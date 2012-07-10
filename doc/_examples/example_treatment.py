import numpy as np

class MyTreat(object):
    def __init__(self, reference=0):
        self.reference = reference

    def code_with_intercept(self, levels):
        return ContrastMatrix(np.eye(len(levels)),
                              ["[My.%s]" % (level,) for level in levels])

    def code_without_intercept(self, levels):
        eye = np.eye(len(levels) - 1)
        contrasts = np.vstack((eye[:self.reference, :],
                               np.zeros((1, len(levels) - 1)),
                               eye[self.reference:, :]))
        suffixes = ["[MyT.%s]" % (level,) for level in
                    levels[:self.reference] + levels[self.reference + 1:]]
        return ContrastMatrix(contrasts, suffixes)
