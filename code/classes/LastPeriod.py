from lag import lag


class LastPeriod:
  def __init__(self):
    return

  @staticmethod
  def gen_factors(X, n_comp = None, threshold = None):
    return 5

  @staticmethod
  def gen_X(X, start):
    if start > 0:
        # remove unused data
        X = X.loc[start-1:]

    # keep only the period before
    # print(X)
    X = lag(X, 1)

    return X

  @staticmethod
  def gen_y(y, start):
    if start > 0:
        # remove unused data
        y = y.loc[start-1,:]

    y = y[1:]

    return y

  def fit(self, X, Y):
    return # do nothing

  def predict(self, X):
    # print(X)
    return X[-1] # the period before

  def predict_train(self, X):
    return self.predict(X)
