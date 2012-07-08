Coding categorical data
=======================

An example of how to write a custom constrast matrix, implementing
'simple' coding as per http://www.ats.ucla.edu/stat/r/library/contrast_coding.htm#SIMPLE

code::
    class Simple(object):
        def _simple_contrast(self, levels):
            nlevels = len(levels)
            contr = -1./nlevels * np.ones((nlevels, nlevels-1))
            contr[1:][np.diag_indices(nlevels-1)] = (nlevels-1.)/nlevels
            return contr

        def code_with_intercept(self, levels):
            contrast = np.column_stack((np.ones(len(levels)),
                                        self._simple_contrast(levels)))
            return ContrastMatrix(contrast, _name_levels("Simp.", levels))

        def code_without_intercept(self, levels):
            contrast = self._simple_contrast(levels)
            return ContrastMatrix(contrast, _name_levels("Simp.", levels[:-1]))

    def test_simple():
        t1 = Simple()
        matrix = t1.code_with_intercept(["a", "b", "c", "d"])
        assert matrix.column_suffixes == ["[Simp.a]","[Simp.b]","[Simp.c]",
                                          "[Simp.d]"]
        assert np.allclose(matrix.matrix, [[1, -1/4.,-1/4.,-1/4.],
                                            [1, 3/4.,-1/4.,-1/4.],
                                            [1, -1/4.,3./4,-1/4.],
                                            [1, -1/4.,-1/4.,3/4.]])
        matrix = t1.code_without_intercept(["a","b","c","d"])
        assert matrix.column_suffixes == ["[Simp.a]","[Simp.b]", "[Simp.c]"]
        assert np.allclose(matrix.matrix, [[-1/4.,-1/4.,-1/4.],
                                            [3/4.,-1/4.,-1/4.],
                                            [-1/4.,3./4,-1/4.],
                                            [-1/4.,-1/4.,3/4.]])
