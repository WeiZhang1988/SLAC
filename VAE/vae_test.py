from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from matplotlib import pyplot as plt 

tf.keras.backend.clear_session()  # For easy reset of notebook state.


class Sampling(layers.Layer):
  """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

  def call(self, inputs):
    z_mean, z_log_var = inputs
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):
  """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""

  def __init__(self,
  			   origin_dim=28,
               latent_dim=2,
               conv2d1_dim=32,
               conv2d2_dim=64,
               dense_dim=16,
               name='encoder',
               **kwargs):
    super(Encoder, self).__init__(name=name, **kwargs)
    self.conv2d1 = layers.Conv2D(conv2d1_dim, 3, activation="relu", strides=2, padding="same",input_shape=(origin_dim,origin_dim,1))
    self.conv2d2 = layers.Conv2D(conv2d2_dim, 3, activation="relu", strides=2, padding="same")
    self.flat = layers.Flatten()
    self.dense = layers.Dense(dense_dim, activation="relu")
    self.dense_mean = layers.Dense(latent_dim)
    self.dense_log_var = layers.Dense(latent_dim)
    self.sampling = Sampling()

  def call(self, inputs):
    x = self.conv2d1(inputs)
    x = self.conv2d2(x)
    x = self.flat(x)
    x = self.dense(x)
    z_mean = self.dense_mean(x)
    z_log_var = self.dense_log_var(x)
    z = self.sampling((z_mean, z_log_var))
    return z_mean, z_log_var, z


class Decoder(layers.Layer):
  """Converts z, the encoded digit vector, back into a readable digit."""

  def __init__(self,
               original_dim=28,
               latent_dim=2,
               dense_dim=7*7*64,
               conv2dt1_dim=64,
               conv2dt2_dim=32,
               name='decoder',
               **kwargs):
    super(Decoder, self).__init__(name=name, **kwargs)
    self.dense = layers.Dense(dense_dim, activation="relu", input_shape=(latent_dim,))
    self.reshape = layers.Reshape((7, 7, conv2dt1_dim))
    self.conv2dt1 = layers.Conv2DTranspose(conv2dt1_dim, 3, activation="relu", strides=2, padding="same")
    self.conv2dt2 = layers.Conv2DTranspose(conv2dt2_dim, 3, activation="relu", strides=2, padding="same")
    self.dense_output = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")

  def call(self, inputs):
    x = self.dense(inputs)
    x = self.reshape(x)
    x = self.conv2dt1(x)
    x = self.conv2dt2(x)
    return self.dense_output(x)


class VariationalAutoEncoder(tf.keras.Model):
  """Combines the encoder and decoder into an end-to-end model for training."""

  def __init__(self,
               origin_dim=28,
               latent_dim=2,
               conv2d1_dim=32,
               conv2d2_dim=64,
               dense1_dim=16,
               dense2_dim=7*7*64,
               conv2dt1_dim=64,
               conv2dt2_dim=32,
               name='autoencoder',
               **kwargs):
    super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
    self.encoder = Encoder()
    self.decoder = Decoder()
    self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
    self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
    self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

  @property
  def metrics(self):
    return [
      self.total_loss_tracker,
      self.reconstruction_loss_tracker,
      self.kl_loss_tracker,
    ]
  def train_step(self,data):
    with tf.GradientTape() as tape:
      z_mean, z_log_var, z = self.encoder(data)
      reconstruction = self.decoder(z)
      reconstruction_loss = tf.reduce_mean(
        tf.reduce_sum(keras.losses.mse(data, reconstruction), axis=(1, 2))
      )
      kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
      kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
      total_loss = reconstruction_loss + kl_loss
    grads = tape.gradient(total_loss, self.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
    self.total_loss_tracker.update_state(total_loss)
    self.reconstruction_loss_tracker.update_state(reconstruction_loss)
    self.kl_loss_tracker.update_state(kl_loss)
    return {
      "loss": self.total_loss_tracker.result(),
      "reconstruction_loss": self.reconstruction_loss_tracker.result(),
      "kl_loss": self.kl_loss_tracker.result(),
    }


"""
## Train the VAE
"""

(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
mnist_digits = np.concatenate([x_train, x_test], axis=0)
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

vae = VariationalAutoEncoder()
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(mnist_digits, epochs=30, batch_size=128)

d=8
z=vae.encoder(np.expand_dims(mnist_digits[d],0))
z=z[0]
print('z',z)
v=vae.decoder(z)
import matplotlib.pyplot as plt
plt.imshow(np.squeeze(mnist_digits[d],-1))
plt.show()
plt.imshow(np.squeeze(v[0],-1))
plt.show()
z1=tf.constant([[-0.5,0.9]])
v1=vae.decoder(z1)
plt.imshow(np.squeeze(v1[0],-1))
plt.show()
