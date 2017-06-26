import numpy as np
import tensorflow as tf
from gen_batches import gen_batches_idx, get_batch


class MLP:
  def __init__(
    self,
    structure,
    learning_rate = 0.001,
    beta = 0.01,
    training_epochs = 3000,
    batch_size = 60,
    display_step = 100,
    debug = False,
    activation = 'relu'
  ):
    self.structure = structure
    self.learning_rate = learning_rate
    self.beta = beta
    self.training_epochs = training_epochs
    self.batch_size = batch_size
    self.display_step = display_step
    self.debug = debug
    self.activation = activation
    self.layers = len(structure)
    self.b2 = None

  def fit(self, X, Y):
        # If several Y are provided (VAR case), use the model on the first one and fit
    # a standard VAR on the others (for h-ahead predictions)
    if Y.shape[1]>1:
      Y = Y.as_matrix()
      Y2 = Y[:, 1:]
      Y = Y[:, 0:1]
      # print(Y)
      self.b2 =  np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,Y2))



    self.initVariables(X)

    # Launch the graph
    sess = tf.Session()
    self.sess = sess
    sess.run(self.init)

    # Training cycle
    for epoch in range(self.training_epochs):
      avg_cost = 0.
      total_batch = int(len(X)/self.batch_size)
      # Loop over all batches
      batches = gen_batches_idx(X, total_batch)
      for batch in batches:
        batch_x, batch_y = get_batch([X, Y], batch)

        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = sess.run([self.optimizer, self.loss], feed_dict={self.x: batch_x,
                                                      self.y: batch_y})
        # Compute average loss
        avg_cost += c / total_batch
      # Display logs per epoch step
      if self.debug and (epoch % self.display_step == 0):
        print("Epoch:", '%04d' % (epoch+1), "cost=", \
          "{:.9f}".format(avg_cost))
    if self.debug:
      print("Optimization Finished!")
    # out_pred = self.pred.eval(feed_dict={x: X_test})
    # print(out_pred)
    # print(np.sqrt(((Y_test - out_pred)*(Y_test - out_pred)).mean()))

  def predict(self, X):
    ret = self.sess.run([self.pred], feed_dict={self.x: X})[0]
    if self.b2 is not None: # predict other variables with a var
      # print(X)
      # print(self.b2)
      # print(np.array(np.dot(X, self.b2)))
      ret = np.array(np.concatenate([ret, np.array(np.dot(X, self.b2), ndmin=1)], axis=1), ndmin=2)

    return ret

  def predict_train(self, X):
    return self.predict(X)

  def multilayer_perceptron(self, x, weights, biases, activation_name):
    if activation_name == 'relu':
      activation = tf.nn.relu
    if activation_name == 'sig':
      activation = tf.nn.sigmoid

    layers = {
      'layer_1': activation(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    }
    for i in range(1, self.layers):
      layer = tf.add(tf.matmul(layers['layer_' + str(i)], weights['h' + str(i+1)]), biases['b' + str(i+1)])
      layers['layer_' + str(i+1)] = activation(layer)

    out_layer = tf.matmul(layers['layer_' + str(self.layers)], weights['out']) + biases['out']

    return out_layer

  def initVariables(self, X):
    n_input = X.shape[1]
    n_hidden_1 = self.structure[0]
    n_hidden_last = self.structure[-1]
    weights = {
      'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
      'out': tf.Variable(tf.random_normal([n_hidden_last, 1]))
    }
    biases = {
      'b1': tf.Variable(tf.random_normal([n_hidden_1])),
      'out': tf.Variable(tf.random_normal([1]))
    }

    for layer, layer_size in enumerate(self.structure[1:]):
      weights['h'+str(layer+2)] = tf.Variable(tf.random_normal([self.structure[layer], layer_size]))
      biases['b'+str(layer+2)] = tf.Variable(tf.random_normal([layer_size]))


    self.x = tf.placeholder("float", [None, n_input])
    self.y = tf.placeholder("float", [None, 1])

    self.pred = self.multilayer_perceptron(self.x, weights, biases, self.activation)

    regularizer = sum([tf.nn.l2_loss(w) for k, w in weights.items() if k != 'out'])

    cost = tf.sqrt(tf.reduce_mean(tf.squared_difference(self.pred, self.y)))

    self.loss = tf.reduce_mean(cost + self.beta * regularizer)
    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    # Initializing the variables
    self.init = tf.global_variables_initializer()
