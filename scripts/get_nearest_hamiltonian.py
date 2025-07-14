import base64
import bz2
import io
import pickle
import sys
import numpy as np
import tensorflow as tf
from Bio import SeqIO
from model.generator import seq_code, read_fasta_as_one_hot_encoded
from model.model import VAE 
import pandas as pd

"""
Usage:
    python get_nearest_hamiltonian.py landscape.pkl sequences.fasta

Note: The sequences in sequences.fasta must be aligned to the model beforehand
"""

def decode_landscape_pkl(pkl_path, model_params):
    with bz2.open(pkl_path, "rb") as f:
         (
        model_title,
        landscape,
        model_seq_len,
        vae_weights,
        training_seqs,
        landscape_seqs
         ) = pickle.load(f)

    # Reconstruct the VAE
    vae = VAE(
        num_aa_types=23,
        dim_latent_vars=2,
        dim_msa_vars=model_seq_len,
        num_hidden_units=model_seq_len * 3,
        activation_func=model_params["activation"],
        regularization=model_params["l2_reg"],
    )
    vae.set_weights(vae_weights)

    return [model_title, landscape, model_seq_len, vae_weights, training_seqs, landscape_seqs], vae

def get_sequence_coords(vae, fasta):
    ds = tf.data.Dataset.from_generator(
        lambda: read_fasta_as_one_hot_encoded(fasta), tf.int8
        ).batch(1)
    z, _, _= vae.encoder.predict(ds)

    headers = []
    seqs = []
    for x in SeqIO.parse(fasta, "fasta"):
        headers.append(x.id)
        seqs.append(str(x.seq))
    return z, (headers, seqs)

import numpy as np

def get_nearest_neighbor_hams(coords, lgl_grid):
    """
    For each point in `coords`, find the nearest (x, y) in `lgl_grid[:, :2]`
    and return the corresponding z values.

    Args:
        coords (np.ndarray): shape (N, 2), query x,y positions
        lgl_grid (np.ndarray): shape (M, 3), columns are x, y, z

    Returns:
        np.ndarray: shape (N,), the z values from nearest neighbors
    """
    # Extract lgl coordinates and values
    lgl_coords = lgl_grid[:, :2]
    z_vals = lgl_grid[:, 2]

    # Output array
    result = np.empty(coords.shape[0])

    for i, point in enumerate(coords):
        # Compute squared Euclidean distance
        dists = np.sum((lgl_coords - point) ** 2, axis=1)
        nearest_idx = np.argmin(dists)
        result[i] = z_vals[nearest_idx]

    return result

def save_results_to_file(path , seq_info, neighbors):
    path = path.split(".")[0]
    table = pd.DataFrame({"headers":seq_info[0], "LGL Hamiltonian": neighbors, "sequence":seq_info[1]})
    table.to_csv(f"{path}_lgl_hamiltonian.csv")
    return

if __name__ == "__main__":

    pkl_path = sys.argv[1]
    fasta_path = sys.argv[2] 
    print("Loading pickle file...")
    model_params = {
        "activation": "relu",  # or "tanh", "sigmoid", etc.
        "l2_reg": 0.001,
    }
    model_info, vae = decode_landscape_pkl(pkl_path, model_params)

    print("Getting sequence coordinates...")
    coords, seq_data = get_sequence_coords(vae, fasta_path)

    print("Getting nearest neighbors")
    nn = get_nearest_neighbor_hams(coords, model_info[1])

    print("Saving to file...")
    save_results_to_file(fasta_path, seq_data, nn)
    print("Done :)")



