import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


class MSReg:
  def __init__(self, k_regimes):
    self.k_regimes = k_regimes
    self.b2 = None


  def fit(self, X, Y):

    if Y.shape[1]>1:
      Y = Y.as_matrix()
      Y2 = Y[:, 1:]
      Y = Y[:, 0:1]
      # print(Y)
      self.b2 =  np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,Y2))

    # if np.array_equal(X[:, 0], np.ones(X[:, 0].shape)): # remove constant as it is added in the model
    #   X = X[:, 1:]


    model =  MarkovRegression(
        endog = np.asanyarray(Y),
        exog = np.asanyarray(X),
        k_regimes = self.k_regimes,
        switching_variance = True
        )
    self.res = model.fit()

  def predict(self, X, h=1):
    """
    Only uses the first row of X (for the next period forecast)
    """
    zeta = np.asanyarray(self.res.smoothed_marginal_probabilities)[-1, :]
    dim = zeta.shape[0]

    def get_param(name):
      return self.res.params[self.res.data.param_names.index(name)]


    P = np.empty([dim, dim])
    # make the probability transition matrix from given coefficients
    for i in range(dim):
      acc = 0
      for j in range(dim-1):
        p = get_param('p[' + str(i) + '->' + str(j) + ']')
        P[j, i] = p
        acc = acc + p
      P[dim-1, i] = 1 - acc

    zeta_new = np.dot(np.linalg.matrix_power(P, h), zeta)
    # print(zeta_new)

    b = np.zeros([dim, X.shape[1] + 1])
    for i in range(dim):
      b[i, 0] = get_param('const[' + str(i) + ']')
      for col in range(1, X.shape[1]+1):
        b[i, col] = get_param('x' + str(col) + '[' + str(i) + ']')

    y_hats = np.dot(b, np.concatenate([[1], X[0, :].T]))

    ret = np.array(np.dot(y_hats, zeta_new), ndmin=1)

    if self.b2 is not None: # predict other variables with a var
      row = X[0, :].T
      ret = np.array(np.concatenate([ret, np.array(np.dot(row, self.b2), ndmin=1)], axis=0), ndmin=2)

    return ret

  def predict_train(self, X):
    ret = np.empty(X[0,:].shape)
    ret[:] = np.NAN
    # return  ret # tmp
    if self.b2 is not None:
      return X[0:1,:]
    return X[0,:] # tmp


