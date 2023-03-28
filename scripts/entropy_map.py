from scipy.stats import entropy
import sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import multiprocessing as mp

"""
Plots average Shannon entropy per amino acid
sys.argv[1]: model path
sys.argv[2]: output path
sys.argv[3]: plot title
"""


def latent_output(model_path):
    vae = tf.keras.models.load_model(model_path)  # import model
    sampling_set = np.linspace(-8, 8, 500)  # create linear subsample space
    a = np.meshgrid(sampling_set, sampling_set)  # create 2D subsample space
    coord = np.vstack(np.array(a).transpose())  # rotate input to proper orientation
    output = vae.decoder.predict(coord)  # get output matrices
    return output, coord


def plot_entropy(entropy, output_path,title):
    plt.imshow(entropy, interpolation='nearest', cmap='inferno', extent=[-8, 8, -8, 8])
    plt.title(title)
    c = plt.colorbar(pad=0.01)
    c.set_label(label="Entropy", fontsize=15)
    plt.savefig(output_path, format="svg")


if __name__ == '__main__':
    mp.freeze_support()
    pool = mp.Pool(mp.cpu_count())
    a, my_coords = latent_output(sys.argv[1])
    num_seq, aa, seq_len = np.shape(a)
    results = pool.map(entropy, a)
    results = pool.map(sum, results)
    results= [value/seq_len for value in results]
    entropy_map = np.reshape(results, (500, 500))
    pool.close()
    plot_entropy(entropy_map, sys.argv[2], sys.argv[3])
