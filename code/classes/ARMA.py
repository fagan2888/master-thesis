import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARMA as sm_ARMA



class ARMA:
  def __init__(self, p, q):
    self.p = p
    self.q = q
    return

  def gen_y(self, y, lags, start):
    return VAR.gen_y(y, lags, start)

  def fit(self, X, Y):
    self.model = sm_ARMA(np.asanyarray(Y), (self.p, self.q)).fit()
    self.start = Y.shape[0]

  def predict(self, X, h=1):
    # print(h)
    if X.shape[0] == self.start: # train
      return self.model.predict()
    else: #test
      # print(self.model.predict(start = self.start, end = self.start+h-1))
      return np.array(self.model.predict(start = self.start, end = self.start+h-1)[-1], ndmin=1)


  def predict_train(self, X):
    return self.predict(X)
