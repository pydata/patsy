import numpy as np
from scipy.stats import norm
from patsy import dmatrices, build_design_matrices

class LM(object):
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
        (new_x,) = build_design_matrices([self._x_design_info.builder],
                                         new_data)
        return np.dot(new_x, self.betas)

    def loglik(self, new_data):
        (new_y, new_x) = build_design_matrices([self._y_design_info.builder,
                                                self._x_design_info.builder],
                                               new_data)
        print new_x
        print self.betas
        new_pred = np.dot(new_x, self.betas)
        sigma = np.sqrt(self.rss / self.nobs)
        return np.log(norm.pdf(new_y, loc=new_pred, scale=sigma))
