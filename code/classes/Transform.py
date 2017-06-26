import pandas as pd
import numpy as np

class Transform:
  def __init__(self, transform_type, cols, params = {}, all_x = False, all_y = False):
    self.cols = cols # True if all, False if none
    self.all_x = all_x
    self.all_y = all_y
    self.params = params
    self.store = False

    if transform_type == 'fun':
      self.transform = self.fun
    elif transform_type == 'log':
      self.transform = self.fun
      self.params = {
        'f': lambda x:  np.log(x),
        'finv': lambda x: np.exp(x)
      }
    elif transform_type == 'standardize':
      self.transform = self.standardize
      self.store = True
    elif transform_type == 'normalize':
      self.transform = self.normalize
      self.store = True
    elif transform_type == 'diff':
      self.transform = self.diff
      self.store = True
      # apply to every col
      self.all_x = True
      self.all_y = True


  def select(self, a, f, is_x):
    all_cols = self.all_x if is_x else self.all_y

    if all_cols == True: # apply to all
      if self.store: return f(a, is_x)
      return f(a)

    cols = list(set(a.columns).intersection(set(self.cols)))
    selected = a[cols]
    not_selected = a.drop(cols, axis=1)

    if self.store: ret = pd.concat([f(selected, is_x), not_selected], axis=1)
    else: ret = pd.concat([f(selected), not_selected], axis=1)

    return ret[a.columns]

  def apply(self, X, y):
    f, _, _ = self.transform()

    # print(self.select(y, f, False))
    return self.select(X, f, True), self.select(y, f, False)

  def reapply(self, a, is_x=True): # for the X_test and to get the correct y_test for the diffs
      _, f, _ = self.transform()
      return self.select(a, f, is_x)

  def unapply(self, y, y_real): # to retransform the y in the original format
    _, _, g = self.transform(y_real)
    return self.select(y, g, False)

  def fun(self, y_real = None):
    """
    Apply an arbitrary function on the dataframe
    """
    return self.params['f'], self.params['f'], self.params['finv']

  def diff(self, y_real = None):
    """
    Integrates each variable of the given degree (and cut the beginning)
    """
    def apply_diffs(col, diffs):
      acc = col
      for _ in range(diffs):
        acc = acc.diff()
      return acc

    self.max_diffs = max([v for k, v in self.params['diffs'].items()])

    def apply(a, is_x):

      # set 0 diffs for the col that are not in the params.diffs
      missing = set(a.columns) - set(self.params['diffs'])
      for val in missing:
        self.params['diffs'][val] = 0

      ret = pd.concat([apply_diffs(a[k], self.params['diffs'][k]) for k in a.columns], axis=1)

      # remove the max_diffs first observations
      ret = ret.iloc[self.max_diffs:]
      return ret

    def reapply(a, is_x):
      if not is_x: # store the real y to generate the undiffed one
        self.y = a
      return apply(a, is_x)

    def unapply(a, is_x):
      start = a.first_valid_index()

      def f(col, diffs):
        col_name = col.columns[0]
        # generate the lists of diffs
        diffs_array = []

        acc = self.y
        for i in range(diffs):
          diffs_array.append(acc)
          acc = acc.diff()

        acc = col
        def g(row, diff):
          idx = row.name
          delta = diffs_array[diffs - diff -1][col_name].loc[idx-1]
          return pd.Series({col_name: (row.ix[0] + delta)})

        for i in range(diffs):
          acc = acc.apply(g, axis=1, args=(i,))

        return acc
      cols = [col for col in a.columns if col in set(self.y.columns).intersection(set(a.dropna(axis=1).columns))]
      # print(set(self.y.columns).intersection(set(a.columns)))
      ret = pd.concat([f(a[[k]], self.params['diffs'][k]) for k in cols], axis=1)
      return ret

    return apply, reapply, unapply

  def standardize(self, y_real = None):
    """
    Center and divide by the std
    """
    def apply(a, is_x):
      mean_val = np.mean(a, axis = 0)
      std_val = np.std(a, axis = 0)
      if is_x:
        self.mean_X = mean_val
        self.std_X = std_val
      else:
        self.mean_y = mean_val
        self.std_y = std_val

      return (a - mean_val) / std_val

    def reapply(a, is_x): # only on X_test
      return (a - self.mean_X) / self.std_X

    def unapply(a, is_x): # only to get y_test_hat
      return a*self.std_y+self.mean_y

    return apply, reapply, unapply


  def normalize(self, y_real = None):
    """
    Change the range to be from -1 to 1
    """
    def apply(a, is_x):
      min_val = np.min(a, axis = 0)
      max_val = np.max(a, axis = 0)

      M = (max_val + min_val)/2
      L = (max_val - min_val)/2
      if is_x:
        self.M_X = M
        self.L_X = L
      else:
        self.M_y = M
        self.L_y = L

      return (a - M) / L

    def reapply(a, is_x): # only on X
      return (a - self.M_X) / self.L_X

    def unapply(a, is_x): # only on y
      return a*self.L_y + self.M_y

    return apply, reapply, unapply
