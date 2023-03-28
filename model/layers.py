import tensorflow as tf
from tensorflow import keras


class Sampling(keras.layers.Layer):
    """Subsample and locate layer by applying gaussian blur"""

    def __call__(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
