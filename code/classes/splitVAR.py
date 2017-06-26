from var_model import VAR
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier

import statsmodels.api as sm


class splitVAR:
  def __init__(self, n_clusters, lags, variables, pred_vars, reg_vars, clustering_type = 'kmeans', proba=False, WLS=False):
    self.n_clusters = n_clusters
    self.clustering_type = clustering_type
    self.lags = lags
    self.vars = variables
    self.pred_vars = pred_vars
    self.reg_vars = reg_vars
    self.proba = proba
    self.WLS = WLS
    self.b2 = None
    return

  def fit(self, X, Y):
    # If several Y are provided (VAR case), use the model on the first one and fit
    # a standard VAR on the others (for h-ahead predictions)
    if Y.shape[1]>1:
      Y = Y.as_matrix()
      Y2 = Y[:, 1:]
      Y = Y[:, 0:1]
      # print(Y)
      self.b2 =  np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,Y2))


    # cluster Y in n_clusters
    # print(np.asmatrix(Y))
    if np.array_equal(X[:, 0], np.ones(X[:, 0].shape)):
      constant = 1
    else:
      constant = 0

    if self.clustering_type == 'kmeans':
      clusteringModel = KMeans(self.n_clusters, n_init=50)
      # cluster_data = np.concatenate([Y, X[:, 0+constant:self.lags+constant]], axis=1)  # clustering with y_t and past values
      cluster_data = Y

    elif self.clustering_type == 'GMM':
      clusteringModel = GaussianMixture(self.n_clusters, n_init=1)
      # cluster_data = np.concatenate([Y, X[:, 0+constant:self.lags+constant]], axis=1)  # clustering with y_t and past values
      cluster_data = Y


    classes = clusteringModel.fit(cluster_data).predict(cluster_data)

    classifier = RandomForestClassifier(max_features=1, n_estimators=50)
    classifier.fit(X, classes)

    self.classifier = classifier

    if (self.WLS):
      weights = classifier.predict_proba(X)

    # run an OLS on each class
    self.bs = np.zeros([self.n_clusters, X.shape[1]])

    for n in range(self.n_clusters):
      if self.WLS: # weighted least squares
        mod_wls = sm.WLS(Y, X, weights=weights[:, n])
        res = mod_wls.fit()
        self.bs[n, :] = res.params

      else:
        X_class = X[classes == n]
        Y_class = Y[classes == n]
        self.bs[n, :] = np.dot(np.linalg.inv(np.dot(X_class.T,X_class)),np.dot(X_class.T,Y_class)).T


  def predict(self, X):
    def f(row):
      if self.proba:
        p = self.classifier.predict_proba(row.reshape(1, -1))
        ret = np.dot(p ,np.dot(self.bs, row))

      else:
        n = self.classifier.predict(row.reshape(1, -1))
        ret = np.dot(self.bs[n[0], :], row)


      if self.b2 is not None: # predict other variables with a var
        ret = np.concatenate([ret, np.dot(row, self.b2)], axis=0)
      return ret

    Y_hat = np.apply_along_axis(f, axis=1, arr=X )
    # Y_hat = np.dot(X, self.b)
    return Y_hat

  def predict_train(self, X):
    return self.predict(X)
