from lag import lag
import numpy as np

class VAR:
  def __init__(self):
    return

  @staticmethod
  def gen_X(X, lags, start):
    # print(X.shape, start)
    if start > 0:
        # remove unused data
        X = X.loc[start - lags:]

    # print(X)
    # generate lags
    X_f = lag(X, lags)

    # train using OLS
    # add intercept
    nrows = X_f.shape[0]
    intercept = np.ones( (nrows,1) )

    X = np.concatenate([intercept, X_f], axis=1)
    return X

  @staticmethod
  def gen_y(y, lags, start):
    # print(y)
    if start > 0:
        # remove unused data
        y = y.loc[start - lags:]

    Y = y[lags-1:]
    return Y

  def fit(self, X, Y):
    self.b = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,Y))

  def predict(self, X):
    Y_hat = np.dot(X, self.b)

    return Y_hat

  def predict_train(self, X):
    Y_hat = np.dot(X, self.b)

    return Y_hat
