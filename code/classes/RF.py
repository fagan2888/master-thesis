import numpy as np
from  sklearn.ensemble import RandomForestRegressor

class RF:
  def __init__(self):
    self.model = RandomForestRegressor(n_estimators = 200)
    return

  def fit(self, X, Y):
    self.model.fit(X, Y)

  def predict(self, X):
    Y_hat = self.model.predict(X)

    return Y_hat

  def predict_train(self, X):
    Y_hat = self.model.predict(X)

    return Y_hat
