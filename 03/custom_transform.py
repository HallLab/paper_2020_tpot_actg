import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator


class AddIntercept(TransformerMixin, BaseEstimator):
    """
    Add a feature called "Intercept" that consists of all 1s
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        """Needed for interface, doesn't actually calculate anything"""
        return self

    def transform(self, X, copy=None):
        """Add an array of ones to be used as the intercept in regression algorithms
        Parameters
        ----------
        X : {array-like, sparse matrix of shape (n_samples, n_features)
            The data that will have an intercept added
        copy : bool, default=None
            Copy the input X or not.
        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features+1)
            Transformed array.
        """
        return np.c_[X, np.ones(X.shape[0])]
