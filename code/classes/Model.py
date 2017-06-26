import pandas as pd
import numpy as np
class Model:

  def __init__(self, model, transforms, gen_x, gen_y, scorefun=None):
    self.model = model
    self.transforms = transforms
    self.gen_x = gen_x
    self.gen_y = gen_y
    self.scorefun = scorefun

  def apply_transforms(self, X, y):
    self.y_cols = y.columns # to restore the cols when we unapply

    ret_x = X
    ret_y = y
    for t in self.transforms:
      ret_x, ret_y = t.apply(ret_x, ret_y)

    self.start = max(ret_y.index) + 1 # to restore the index when we unapply
    return ret_x, ret_y

  def reapply_transforms(self, X):
    ret_x = X
    for t in self.transforms:
      ret_x = t.reapply(ret_x)

    return ret_x

  def unapply_transforms(self, y, y_real, start = None):
    if len(y.shape) > 1:
      cols_slice = y.shape[1]
    else:
      cols_slice = 1
    if not isinstance(y, pd.DataFrame):
      y = np.asanyarray(y)
      if start is None:
        offset = len(y_real.index) - y.shape[0]
        start = y_real.first_valid_index() + offset
      y = pd.DataFrame(y, index=range(start, start+y.shape[0]), columns=self.y_cols[0:cols_slice])

    acc = y_real
    for t in self.transforms:
      acc = t.reapply(acc, False)

    ret_y = y
    for t in reversed(self.transforms):
      ret_y = t.unapply(ret_y, y_real)

    return ret_y

  def train(self, X, y):
    self.start = max(X.index)+1
    X, y = self.apply_transforms(X, y)
    self.model.fit(self.gen_x(X, 0), self.gen_y(y, 0))

  def train_var(self, X, y):
    self.start = max(X.index)+1
    X, y = self.apply_transforms(X, y)
    self.model.fit(self.gen_x(X, 0), self.gen_y(y, 0))


  def predict(self, X, h=1, prediction_type = 'direct'):
    X = self.reapply_transforms(X)

    if prediction_type == 'direct':
      # print(self.model.predict(self.gen_x(X, self.start-h+1)))
      return self.model.predict(self.gen_x(X, self.start-h+1))
    elif prediction_type == 'dynamic-ar':
      return self.model.predict(self.gen_x(X, self.start-h+1), h) # X never used?

  def predict_ar(self, X, h=1, prediction_type = 'dynamic'):
    cols = X.columns
    X = self.reapply_transforms(X)
    X = X[cols]

    for i in range(h-1):
      if (prediction_type == 'dynamic-ar'):
        new_row = self.model.predict(self.gen_x(X, X.last_valid_index()+1), i+1)
      else:
        new_row = self.model.predict(self.gen_x(X, X.last_valid_index()+1))
      # print(new_row)
      X.loc[X.last_valid_index()+1] = pd.Series(new_row[0], index=list(X.columns) )
      # print(X)
    if (prediction_type == 'dynamic-ar'):
      return self.model.predict(self.gen_x(X, X.last_valid_index()+1), h)
    else:
      return self.model.predict(self.gen_x(X, X.last_valid_index()+1))

  def predict_train(self, X):
    X = self.reapply_transforms(X)

    return self.model.predict_train(self.gen_x(X, 0))

  def test(self, y_hat, y_real):
    y_pred = self.unapply_transforms(y_hat, y_real)
    return self.scorefun(y_pred, y_real), y_pred

  def compute_errors(self, y_hat, y_test):
    start = y_hat.first_valid_index()
    errors = y_hat - y_test[start-1:]
    return errors

  def expanding_window_old(self, X, y, start, hs=[1], prediction_type='direct'):
    ret = {}
    for h in hs:
      predictions = []
      train_scores = []
      for i in range(start, X.last_valid_index()+1):
        prediction, score_train = self.predict_at(X, y, i, prediction_type, h)
        predictions.append(prediction)
        train_scores.append(score_train)
        if i % 10 == 0:
          print('step: ' + str(i))

      predictions = pd.concat(predictions)
      errors = self.compute_errors(predictions, y)
      ret[h] = (self.scorefun(predictions, y), predictions, errors, train_scores)
    return ret

  def expanding_window(self, X, y, start, hs=[1], prediction_type='dynamic'):
    if prediction_type == 'direct':
      return self.expanding_window_old(X, y, start, hs, prediction_type)

    all_predictions = {h: pd.DataFrame() for h in hs}
    all_train_errors = pd.DataFrame(columns = ['error'], index = range(start-max(hs), X.last_valid_index()+1))

    for t in range(start-max(hs), X.last_valid_index()+1):
      if t % 5 == 0:
        print('step: ' + str(t))

      predictions, score_train = self.predict_from(X, y, t, prediction_type, hs, start)
      all_train_errors.loc[t] = pd.Series({'error': score_train})
      for h in predictions.keys():
        all_predictions[h] = all_predictions[h].append(predictions[h])

    ret = {}
    for h in hs:
      errors = all_train_errors.loc[start-h:X.last_valid_index()-h].copy()
      errors.index += h
      ret[h] =  (
                  self.scorefun(all_predictions[h], y),
                  all_predictions[h],
                  self.compute_errors(all_predictions[h], y),
                  errors
                )

    return ret

  def predict_from(self, X, y, t, prediction_type, hs, start):
    # split the data at t
    first = X.first_valid_index()

    X_train = X.loc[first:t-1]

    if X.shape[1] >1: # VAR
      y_train = X.loc[first+1:t]
    else:
      y_train = y.loc[first+1:t]

    # train
    if X.shape[1] >1: # VAR
      self.train_var(X_train, y_train)
      pred_train = self.predict_train(X_train)[:,0]
    else :
      self.train(X_train, y_train)
      pred_train = self.predict_train(X_train)

    score_train, _ = self.test(pred_train, y_train)

    # print(X_train)
    predictions = {}
    for h in hs:
      if t+h >= start and t+h <= X.last_valid_index():
        X_test = X.loc[first:t]
        y_test = y.loc[t+h:t+h]
        # y_test = y.loc[first+1:t+h]

        if prediction_type == 'dynamic' or (prediction_type == 'dynamic-ar' and X.shape[1] >1):
          pred_test = self.predict_ar(X_test, h, prediction_type)#[:, 0:1]
        else:
          pred_test = self.predict(X_test, h, prediction_type)
        # print(pred_test)

        prediction = self.unapply_transforms(pred_test, y_test, t+h)
        predictions[h] = prediction

    return predictions, score_train


  def predict_at(self, X, y, start, prediction_type='direct', h=1):
    # split the data at start
    # X_train = X.loc[0:start-1]
    # X_test = X.loc[0:start]

    # y_train = y.loc[0:start-1]
    # y_test = y.loc[0:start]
    # print(h)

    # split the data at start
    first = X.first_valid_index()

    if prediction_type == 'direct':
      X_train = X.loc[first:start-2*h]
      X_test = X.loc[first:start-h]

      y_train = y.loc[first+h:start-h]
      y_test = y.loc[first+h:start]

    elif prediction_type == 'dynamic' or prediction_type == 'dynamic-ar':
      X_train = X.loc[first:start-h-1]
      X_test = X.loc[first:start-h]

      if prediction_type == 'dynamic' and X.shape[1] >1: # VAR
        y_train = X.loc[first+1:start-h]
      else:
        y_train = y.loc[first+1:start-h]

      y_test = y.loc[first+1:start-h+1]

    # print(start)
    # print(X_train)
    # print(y_train)

    # print(X_test)
    # print(y_test)

    # print(X_train.shape, y_train.shape)

    if prediction_type == 'dynamic' and X.shape[1] >1: # VAR
      self.train_var(X_train, y_train)
      pred_train = self.predict_train(X_train)[:,0]
    else :
      self.train(X_train, y_train)
      pred_train = self.predict_train(X_train)

    if prediction_type == 'dynamic':
      pred_test = self.predict_ar(X_test, h)#[:, 0:1]
    else:
      pred_test = self.predict(X_test, h, prediction_type)


    # print(pred_test)
    prediction = self.unapply_transforms(pred_test, y_test, start)
    # print(prediction)

    score_train, _ = self.test(pred_train, y_train)

    # score_test, out_pred = self.test(pred_test, y_test)
    return prediction, score_train


