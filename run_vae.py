import datetime
import sys

import tensorflow as tf
from tensorflow import keras

from model.generator import get_fasta_file_dimensions, read_fasta_as_one_hot_encoded
from model.model import VAE

# ARGV 1 = FASTA file path
# ARGV 2 = model save path
# ARGV 3 =log directory

log_dir = sys.argv[3]+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
fasta_in = sys.argv[1]
model_out = sys.argv[2]

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='loss',patience=10)
save_best_model = tf.keras.callbacks.ModelCheckpoint(model_out,monitor='loss',model='min',save_best_only=True)

BATCH_SIZE = 16
EPOCHS = 1000

num_sequences, seq_len = get_fasta_file_dimensions(fasta_in)

print(
   f"Sequence length: {seq_len} \n Number sequences: {num_sequences} \n steps_per_epoch={num_sequences // BATCH_SIZE}")

vae = VAE(num_aa_types=23, dim_latent_vars=2, dim_msa_vars=seq_len, num_hidden_units=seq_len*3, activation_func='relu', regularization=1e-4)
vae.compute_output_shape(input_shape=(None, 23 * seq_len))

# Wrap generator in tf.Dataset.
dtypes = tf.int8  # datatype for return value of read_fasta_as_one_hot_encoded np.int8 -> tf.int8
ds = tf.data.Dataset.from_generator(lambda: read_fasta_as_one_hot_encoded(fasta_in), dtypes)
ds = ds.shuffle(1000)  # Choose a random sequence from a buffer of 1000 sequences.
ds = ds.batch(BATCH_SIZE)

# Create model and fit.
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(ds,
        epochs=EPOCHS,
        # validation_data=(test_msa, test_msa),
        callbacks=[earlystopping,save_best_model]
        )
vae.save(model_out)