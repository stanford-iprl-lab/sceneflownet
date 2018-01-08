import tensorflow as tf

# Thanks, https://github.com/tensorflow/tensorflow/issues/4079
def LeakyReLU(x, leak=0.1, name="lrelu"):
  with tf.variable_scope(name):
    f1 = 0.5 * (1.0 + leak) 
    f2 = 0.5 * (1.0 - leak)
    return f1 * x + f2 * abs(x)


