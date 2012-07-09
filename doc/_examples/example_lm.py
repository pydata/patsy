import numpy as np
from scipy.stats import norm
from charlton import dmatrices, build_design_matrices

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
                   "  Regression (beta) coefficients:"
                   % (self._y_design_info.describe(),
                      self._x_design_info.describe()))
        for name, value in zip(self._x_design_info.column_names, self.betas):
            summary += "    %s: %s\n" % (name, value)

    def predict(self, new_data):
        (new_x,) = build_design_matrices([self._x_design_info.builder],
                                         new_data)
        return np.dot(new_x, self.betas)

    def loglik(self, new_data):
        (new_y, new_x) = build_design_matrices([self._x_design_info.builder,
                                                self._y_design_info.builder],
                                               new_data)
        new_pred = np.dot(new_x, self.betas)
        sigma = np.sqrt(self.rss / self.nobs)
        return norm.logpdf(new_y, loc=new_pred, scale=sigma)

    # XX: type-I ANOVA to demonstrate terms?
    # XX: glht to demonstrate linear_constraint()?
