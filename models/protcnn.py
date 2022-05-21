import tensorflow as tf
import tensorflow.keras as keras

from functools import partial
import numpy as np

from utils import aa_letters, luxa_seq
from utils.metrics import aa_acc
from utils.data_loaders import right_pad, to_one_hot
from utils.decoding import _decode_ar, _decode_nonar, batch_temp_sample

nchar = len(aa_letters)  # = 21

class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class BaseProtVAE(keras.Model):
    def __init__(self, encoder=None, decoder=None, **kwargs):
        super(BaseProtVAE, self).__init__()

        if encoder is not None:
            self.encoder = encoder
        if decoder is not None:            
            self.decoder = decoder
            
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.acc_tracker = keras.metrics.Mean(name="categorical_acc")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.acc_tracker,
        ]

    def train_step(self, data):
        data=data[0] # since alexhh's train_gen return (data, data)
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(data)
            z = Sampling()([z_mean, z_log_var])

            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.categorical_crossentropy(data, reconstruction), axis=(1) # Fix dimension from Keras' example
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss

            # print(data, reconstruction)
            acc = aa_acc(data, reconstruction)
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.acc_tracker.update_state(acc)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "categorial_acc": self.acc_tracker.result(),
        }
        
    def prior_sample(self, n_samples=1, mean=0, stddev=1,
                     remove_gaps=False, batch_size=5000):
        if n_samples > batch_size:
            x = []
            total = 0
            while total< n_samples:
                this_batch = min(batch_size, n_samples - total)
                z_sample = mean + stddev * np.random.randn(this_batch, self.latent_dim)
                x += self.decode(z_sample, remove_gaps=remove_gaps)
                total += this_batch
        else:
            z_sample = mean + stddev * np.random.randn(n_samples, self.latent_dim)
            x = self.decode(z_sample, remove_gaps=remove_gaps)
        return x

    def decode(self, z, remove_gaps=False, sample_func=None, conditions=None):
        return _decode_nonar(self.decoder, z, remove_gaps=remove_gaps, conditions=conditions)

    def generate_variants_luxA(self, num_samples, posterior_var_scale=1., temperature=0.,
                               solubility_level=None,remove_gaps=True):

        luxa_oh = to_one_hot(right_pad([luxa_seq], self.encoder.input_shape[1]))
        luxa_oh = np.repeat(luxa_oh, num_samples, axis=0)
        orig_conds = np.repeat(np.array([1,0,0]).reshape((1,3)), num_samples, axis=0)
        inputs = luxa_oh if solubility_level is None else [luxa_oh, orig_conds]

        # luxa_zmean, luxa_zvar, luxa_z = self.stochastic_E.predict(inputs)
        z_mean, z_log_var = self.encoder(inputs)
        luxa_z = Sampling()([z_mean, z_log_var])

        print(luxa_z.shape)

        # if posterior_var_scale != 1.:
        #     luxa_z = np.sqrt(posterior_scale*luxa_zvar)*np.random.randn(*luxa_zmean.shape) + luxa_zmean

        sample_func = None
        if temperature > 0:
            sample_func = partial(batch_temp_sample, temperature=temperature)
        target_conds = None if solubility_level is None else luxa_batch_conds(num_samples, solubility_level)
        return self.decode(luxa_z, remove_gaps=remove_gaps, sample_func=sample_func,
                           conditions=target_conds)