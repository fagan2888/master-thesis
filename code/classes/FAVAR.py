from var_model import VAR
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

class FAVAR:
  def __init__(self, factors):
    self.factors = factors
    return

  def addFactors(self, X, n):
    factors = np.asanyarray(self.factors.loc[X.index])
    # remove factors that have nulls values
    # factors = factors.ix[:, factors.isnull().sum(axis=0)==0]

    factors = factors[:, ~np.any(np.isnan(factors), axis=0)]

    # center and standardise
    f_mean = factors.mean()
    f_std = factors.std()
    factors = (factors - f_mean)/f_std

    # fit PCA projection
    self.pca = PCA(n_components=n)
    self.pca.fit(factors)

    f = pd.DataFrame(self.pca.transform(factors), index = X.index)
    f.columns = ['f_' + str(i) for i in range(n)]

    # add factors after the usual columns
    X = pd.concat([X, f], axis=1)
    return X

  # rest is not used anymore

  def gen_X(self, X, lags, start):
    factors = np.asanyarray(self.factors.loc[X.index])

    # remove factors that have nulls values
    # factors = factors.ix[:, factors.isnull().sum(axis=0)==0]

    # print(factors[:,~np.all(np.isnan(factors), axis=0)])
    factors = factors[:, ~np.any(np.isnan(factors), axis=0)]
    # print(np.isnan(factors))

    # center and standardise
    f_mean = factors.mean()
    f_std = factors.std()
    factors = (factors - f_mean)/f_std

    if start == 0: # train
      # fit PCA projection

      self.pca = PCA(n_components=self.n_pc)
      self.pca.fit(factors)

    # else, test: reuse factor projection
    f = pd.DataFrame(self.pca.transform(factors), index = X.index)

    # add factors after the usual columns
    X = pd.concat([X, f], axis=1)
    return VAR.gen_X(X, lags, start)

  def gen_y(self, y, lags, start):
    return VAR.gen_y(y, lags, start)

  def fit(self, X, Y):
    self.b = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,Y))

  def predict(self, X):
    Y_hat = np.dot(X, self.b)

    return Y_hat

  def predict_train(self, X):
    return self.predict(X)
