import numpy as np
from patsy import dmatrices, build_design_matrices

class LM(object):
    """An example ordinary least squares linear model class, analogous to R's
    lm() function. Don't use this in real life, it isn't properly tested."""
    def __init__(self, formula_like, data={}):
        y, x = dmatrices(formula_like, data, 1)
        self.nobs = x.shape[0]
        self.betas, self.rss, _, _ = np.linalg.lstsq(x, y)
        self._y_design_info = y.design_info
        self._x_design_info = x.design_info

    def __repr__(self):
        summary = ("Ordinary least-squares regression\n"
                   "  Model: %s ~ %s\n"
                   "  Regression (beta) coefficients:\n"
                   % (self._y_design_info.describe(),
                      self._x_design_info.describe()))
        for name, value in zip(self._x_design_info.column_names, self.betas):
            summary += "    %s:  %0.3g\n" % (name, value[0])
        return summary

    def predict(self, new_data):
        (new_x,) = build_design_matrices([self._x_design_info],
                                         new_data)
        return np.dot(new_x, self.betas)

    def loglik(self, new_data):
        (new_y, new_x) = build_design_matrices([self._y_design_info,
                                                self._x_design_info],
                                               new_data)
        new_pred = np.dot(new_x, self.betas)
        sigma2 = self.rss / self.nobs
        # It'd be more elegant to use scipy.stats.norm.logpdf here, but adding
        # a dependency on scipy makes the docs build more complicated:
        Z = -0.5 * np.log(2 * np.pi * sigma2)
        return Z + -0.5 * (new_y - new_x) ** 2/sigma2
