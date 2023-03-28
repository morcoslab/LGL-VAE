from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from model.generator import seq_code, read_fasta_as_one_hot_encoded
from dca.dca_class import dca
import sys
import pickle
import tensorflow as tf


### Requires src-python-mfdca package from EIL github ###
#
# Produces MSA of grid sequences, saved as 'grid_msa.fasta'
# Produces numpy array of form [x,y,Hamiltonian], saved as grid_dataset.pkl
# Produces pixel rendered plot saved as pixel_plot.png
# sys.argv[1] = trained model
# sys.argv[2] = MSA used to train model
# sys.argv[3] = sequence of interest use to calculate deltas
# sys.argv[4] = output fasta with coordinates and sequences
# sys.argv[5] = output pickle for delta hamiltonian grid
# sys.argv[6] = x min
# sys.argv[7] = x max
# sys.argv[8] = y min
# sys.argv[9] = y max
# sys.argv[10] = resolution
# sys.argv[11] = bool for whether delta is calculated using the Hamiltonian of the
#                actual sequence or the reconstructed sequence. True = actual seq
#                False = reconstructed seq


def get_key(val):
    for key, value in seq_code.items():
        if val == value:
            return key


def return_sequence(latent_output):
    seq = ''.join(get_key(x) for x in np.argsort(latent_output, axis=0)[-1, :])
    return seq


def make_grid_msa(model_path, output_path, x_resolution, y_resolution):
    trained_model = load_model(model_path)
    sampling_set_x = np.linspace(x_resolution[0], x_resolution[1], x_resolution[2])
    sampling_set_y = np.linspace(y_resolution[0], y_resolution[1], y_resolution[2])
    a = np.meshgrid(sampling_set_x, sampling_set_y)
    coord = np.vstack(np.array(a).transpose())
    with open(output_path, 'w') as fd:
        latent_output = trained_model.decoder.predict(coord)
        sequences = [return_sequence(seq_mat) for seq_mat in latent_output]
        for idx_seq, (x, y) in enumerate(coord):
            fd.writelines('> ' + str(x) + ' ' + str(y) + '\n')
            fd.writelines(sequences[idx_seq] + '\n')
    fd.close()
    return coord


def get_delta_hamiltonian(wt_fasta, dca_fasta_in, model_grid_in, x_resolution, y_resolution, coords_for_pkl):
    mfdcamodel = dca(dca_fasta_in)
    mfdcamodel.mean_field()
    wt_hamiltonian = mfdcamodel.compute_Hamiltonian(wt_fasta)[0]
    grid_hamiltonians, _ = mfdcamodel.compute_Hamiltonian(model_grid_in)
    delta_hamiltonian = grid_hamiltonians - wt_hamiltonian
    grid_for_plotter = np.zeros((coords_for_pkl.shape[0], 3))
    grid_for_plotter[:, :2] = coords_for_pkl
    grid_for_plotter[:, 2] = delta_hamiltonian

    delta_hamiltonian = delta_hamiltonian.reshape(x_resolution[2], y_resolution[2])
    return delta_hamiltonian, grid_for_plotter


def get_delta_hamiltonian_reconstructed(model_path, wt_fasta, dca_fasta_in, model_grid_in, x_resolution,
                                        y_resolution, coords_for_pkl):
    trained_model = load_model(model_path)
    ds = tf.data.Dataset.from_generator(lambda: read_fasta_as_one_hot_encoded(wt_fasta), tf.int8)
    ds = ds.batch(1)
    z_m, z_s, z_out = trained_model.encoder.predict(ds)
    reconstruction_mat = trained_model.decoder(z_m)
    reconstruction_seq = return_sequence(reconstruction_mat[0])

    recon_file_name = wt_fasta.split(".")[0]+"_reconstructed.fasta"
    with open(recon_file_name, 'w') as fd:
        fd.writelines('> wt reconstruction' + '\n')
        fd.writelines(str(reconstruction_seq) + '\n')
    fd.close()

    mfdcamodel = dca(dca_fasta_in)
    mfdcamodel.mean_field()
    wt_hamiltonian = mfdcamodel.compute_Hamiltonian(recon_file_name)
    grid_hamiltonians, _ = mfdcamodel.compute_Hamiltonian(model_grid_in)
    delta_hamiltonian = grid_hamiltonians - wt_hamiltonian[0]
    grid_for_plotter = np.zeros((coords_for_pkl.shape[0], 3))
    grid_for_plotter[:, :2] = coords_for_pkl
    grid_for_plotter[:, 2] = delta_hamiltonian

    delta_hamiltonian = delta_hamiltonian.reshape(x_resolution[2], y_resolution[2])
    return delta_hamiltonian, grid_for_plotter


def plot_hamil_latent(hamil, title, resolution):
    # generate 2 2d grids for the x & y bounds
    a, b = np.meshgrid(np.linspace(resolution[0], resolution[1], resolution[2]),
                       np.linspace(resolution[0], resolution[1], resolution[2]))

    fig, ax = plt.subplots()
    c = ax.pcolormesh(b, a, hamil, cmap='jet')
    ax.set_title(title)
    fig.colorbar(c, ax=ax)
    plt.show()


######
# MAIN #
########

calc_method = sys.argv[11].lower()
pixels = int(sys.argv[10])  # N^2 grid of sequences will be produced
x_axis_min = float(sys.argv[6])
x_axis_max = float(sys.argv[7])
x_resolution = [x_axis_min, x_axis_max, pixels]
y_axis_min = float(sys.argv[8])
y_axis_max = float(sys.argv[9])
y_resolution = [y_axis_min, y_axis_max, pixels]

model = sys.argv[1]
full_alignment = sys.argv[2]
wt_fasta = sys.argv[3]
output_path = sys.argv[4]
output_pkl_path = sys.argv[5]
coordinates = make_grid_msa(model, output_path, x_resolution, y_resolution)

if calc_method == "true":
    hamil_mat, grid_for_pkl = get_delta_hamiltonian(wt_fasta, full_alignment, output_path, x_resolution, y_resolution,
                                                    coordinates)
elif calc_method == "false":
    hamil_mat, grid_for_pkl = get_delta_hamiltonian_reconstructed(model, wt_fasta, full_alignment, output_path,
                                                                  x_resolution, y_resolution, coordinates)
else:
    exit("Calculation method is not a boolean value")

# save pickle file for plot_model.py
pickle.dump(grid_for_pkl, open(output_pkl_path, 'wb'))
