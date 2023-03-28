import tensorflow as tf
from tensorflow import keras

from model.layers import Sampling


class VAE(keras.Model):
    def __init__(self, num_aa_types, dim_latent_vars, dim_msa_vars, num_hidden_units, activation_func, regularization, **kwargs):
        super(VAE, self).__init__(**kwargs)
        ## num of amino acid types
        self.num_aa_type = num_aa_types
        ## dimension of latent space
        self.dim_latent_vars = dim_latent_vars
        ## dimension of binary representation of sequences
        self.dim_msa_vars = dim_msa_vars
        ## num of hidden neurons in encoder and decoder networks
        self.num_hidden_units = num_hidden_units
        self.activation = activation_func
        self.reg = regularization
        self.multiclass = False
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def get_encoder(self):
        input_layer = tf.keras.Input(shape=(self.num_aa_type * self.dim_msa_vars,))
        hidden_layer = tf.keras.layers.Dense(self.num_hidden_units, activation=self.activation,
                                            kernel_regularizer=tf.keras.regularizers.L2(self.reg))(input_layer)
        z_mean = tf.keras.layers.Dense(self.dim_latent_vars, name="z_mean")(hidden_layer)
        z_sigma = tf.keras.layers.Dense(self.dim_latent_vars, name="z_sigma")(hidden_layer)
        latent_space = Sampling()([z_mean, z_sigma])
        encoder = tf.keras.Model(input_layer, [z_mean, z_sigma, latent_space], name='encoder')
        return encoder

    def get_decoder(self):
        latent_input = tf.keras.Input(shape=(self.dim_latent_vars,), name='z_sampling')
        reconstructed_hidden_layer = tf.keras.layers.Dense(self.num_hidden_units, activation=self.activation,
                                                    kernel_regularizer=tf.keras.regularizers.L2(self.reg))(latent_input)
        output_layer = tf.keras.layers.Dense(self.num_aa_type * self.dim_msa_vars)(reconstructed_hidden_layer)
        reshape_layer = tf.keras.layers.Reshape((self.num_aa_type, self.dim_msa_vars,))(output_layer)
        activated_output = tf.keras.layers.Softmax(axis=1)(reshape_layer)
        decoder = tf.keras.Model(latent_input, activated_output, name='decoder')
        return decoder

    def call(self, data):
        _, _, z = self.encoder(data)
        y_pred = self.decoder(z)
        return y_pred


    def evaluate(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_sigma, z = self.encoder(data)
            output = self.decoder(z)
            reshaped = tf.keras.layers.Reshape((self.num_aa_type * self.dim_msa_vars,))(output)
            reconstruction_loss = tf.keras.losses.binary_crossentropy(data, reshaped)
            reconstruction_loss *= self.num_aa_type * self.dim_msa_vars

            kl_loss = 1 + z_sigma - keras.backend.square(z_mean) - keras.backend.exp(z_sigma)
            kl_loss = keras.backend.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            # kl_loss *= self.beta
            vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)
        return vae_loss, keras.backend.mean(reconstruction_loss), keras.backend.mean(kl_loss)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_sigma, z = self.encoder(data)
            output = self.decoder(z)
            reshaped = tf.keras.layers.Reshape((self.num_aa_type * self.dim_msa_vars,))(output)
            reconstruction_loss = tf.keras.losses.binary_crossentropy(data, reshaped)
            reconstruction_loss *= self.num_aa_type * self.dim_msa_vars

            kl_loss = 1 + z_sigma - keras.backend.square(z_mean) - keras.backend.exp(z_sigma)
            kl_loss = keras.backend.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            # kl_loss *= self.beta
            vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)
        grads = tape.gradient(vae_loss, self.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(vae_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss"               : self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss"            : self.kl_loss_tracker.result(),
        }
