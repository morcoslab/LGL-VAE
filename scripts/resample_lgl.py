import base64
import bz2
import io
import pickle
import sys
import numpy as np
import tensorflow as tf
from Bio import SeqIO
from tensorflow.keras.models import load_model
from model.generator import seq_code, read_fasta_as_one_hot_encoded
from model.model import VAE  # Assuming you have this class defined somewhere
from dca.dca_class import dca
import os

"""
Usage:
    python resample_lgl.py landscape.pkl [pixel_density=400] [x_min=-8] [x_max=8] [y_min=-8] [y_max=8]
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

    # Decode training FASTA
    # decoded_train = base64.b64decode(training_seqs)
    # text_train = io.StringIO(decoded_train.decode("utf-8"))
    # train_fasta = list(SeqIO.parse(text_train, "fasta"))

    return [model_title, landscape, model_seq_len, vae_weights, training_seqs, landscape_seqs], vae

def get_axis_values(model, training_seqs) -> int:
    # Convert sequences to one-hot using your generator
    def generator():
        for record in training_seqs:
            yield read_fasta_as_one_hot_encoded(record.seq)

    ds = tf.data.Dataset.from_generator(generator, output_types=tf.int8).batch(1000)
    _, _, zed = model.encoder.predict(ds)
    largest_value = abs(max(zed.min(), zed.max()))
    return int(np.ceil(largest_value))

def get_key(val):
    for key, value in seq_code.items():
        if val == value:
            return key

def return_sequence(latent_output):
    seq = ''.join(get_key(x) for x in np.argsort(latent_output, axis=0)[-1, :])
    return seq

def make_grid_msa(trained_model, resolution, batch_size=10000):
    x_sampling_set = np.linspace(resolution[1], resolution[2], resolution[0])
    y_sampling_set = np.linspace(resolution[3], resolution[4], resolution[0])
    a = np.meshgrid(x_sampling_set, y_sampling_set)
    coord = np.vstack(np.array(a).reshape(2, -1).T)

    lines = []
    for batch_idx in range(0, coord.shape[0], batch_size):
        z_input = coord[batch_idx:batch_idx + batch_size]
        latent_output = trained_model.decoder.predict(z_input)
        sequences = [return_sequence(seq_mat) for seq_mat in latent_output]
        for idx_seq, (x, y) in enumerate(z_input):
            lines.append(f'> {x} {y}\n')
            lines.append(sequences[idx_seq] + '\n')

    fasta_str = ''.join(lines)
    fasta_bytes = fasta_str.encode('utf-8')
    return coord, fasta_bytes


def get_hamiltonian(training_fasta_bytes: bytes, landscape_fasta_bytes: bytes, coords_for_pkl: np.array) -> np.array:
    
    training_fasta_bytes = base64.b64decode(training_fasta_bytes)
    training_fasta_path = "training_tmp.fasta"
    landscape_fasta_path = "landscape_tmp.fasta"

    try:
        # Write to real files
        with open(training_fasta_path, "wb") as f:
            f.write(training_fasta_bytes)
        with open(landscape_fasta_path, "wb") as f:
            f.write(landscape_fasta_bytes)

        # Run your own DCA model 
        mfdcamodel = dca(training_fasta_path)
        mfdcamodel.mean_field()

        grid_hamiltonians, _ = mfdcamodel.compute_Hamiltonian(landscape_fasta_path)

        # Combine with coordinates
        output_grid = np.zeros((coords_for_pkl.shape[0], 3))
        output_grid[:, :2] = coords_for_pkl
        output_grid[:, 2] = grid_hamiltonians

    finally:
        # Clean up files
        if os.path.exists(training_fasta_path):
            os.remove(training_fasta_path)
        if os.path.exists(landscape_fasta_path):
            os.remove(landscape_fasta_path)

    return output_grid

def save_updated_pkl(old_path, model_info, landscape_fasta_bytes, new_grid, resolution):
    """
    Save updated .pkl file with Hamiltonian landscape and new grid sampling.
    
    Args:
        old_path (str): Path to the original .pkl file.
        model_info (tuple): Tuple of original model metadata.
        landscape_fasta_bytes (bytes): Binary FASTA string of resampled grid.
        new_grid (np.array): Coordinates and Hamiltonian values.
        resolution (list): [pixel_density, x_min, x_max, y_min, y_max]
    """
    # Unpack model info
    model_title, _, model_seq_len, vae_weights, training_fasta_bytes, _ = model_info

    # Base64 encode FASTA content
    encoded_landscape_fasta = base64.b64encode(landscape_fasta_bytes).decode("utf-8")

    # Updated data tuple
    updated_obj = (
        model_title,
        new_grid,
        model_seq_len,
        vae_weights,
        training_fasta_bytes,
        encoded_landscape_fasta
    )

    # Construct new file name based on resolution
    base, _ = os.path.splitext(old_path)
    pixel_density, x_min, x_max, y_min, y_max = resolution
    new_filename = (
        f"{base}_px{pixel_density}_x_{x_min}_{x_max}_y_{y_min}_{y_max}"
        .replace(".", "p")  # Clean up decimal points for safe filenames
    )
    new_filename +=".pkl"
    # Save as new file with bz2 compression
    with bz2.open(new_filename, "wb") as f:
        pickle.dump(updated_obj, f)

    print(f"[✔] Updated pickle file saved as: {new_filename}")

if __name__ == "__main__":
    pkl_path = sys.argv[1]
    pixel_density = int(sys.argv[2]) if len(sys.argv) > 2 else 400

    # Optional axis bounds
    x_min = float(sys.argv[3]) if len(sys.argv) > 3 else -8
    x_max = float(sys.argv[4]) if len(sys.argv) > 4 else 8
    y_min = float(sys.argv[5]) if len(sys.argv) > 5 else -8
    y_max = float(sys.argv[6]) if len(sys.argv) > 6 else 8

    # You must define this to match how the model was trained
    model_params = {
        "activation": "relu",  # or "tanh", "sigmoid", etc.
        "l2_reg": 0.001,
    }

    # Decode and restore VAE + training data
    print("Loading pickle file...")
    model_info, vae = decode_landscape_pkl(pkl_path, model_params)

    resolution = [pixel_density, x_min, x_max, y_min, y_max]
    print("Generating new grid...")
    coordinates, sampled_fasta_bytes = make_grid_msa(
        trained_model=vae,
        resolution=resolution
    )
    print("Calculating Hamiltonians...")
    new_ham_grid = get_hamiltonian(model_info[4], sampled_fasta_bytes, coordinates)

    save_updated_pkl(pkl_path, model_info, sampled_fasta_bytes, new_ham_grid, resolution)


