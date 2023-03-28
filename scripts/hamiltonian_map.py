import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from model.generator import seq_code, read_fasta_as_one_hot_encoded
from dca.dca_class import dca
import time
import pickle

### Requires src-python-mfdca package from EIL github ###
#
# Produces MSA of grid sequences, saved as 'grid_msa.fasta'
# Produces numpy array of form [x,y,Hamiltonian], saved as grid_dataset.pkl
# Produces pixel rendered plot saved as pixel_plot.png
#

# parameters to change
model = 'trained_models/laao' # name of the model you're using
output_path = 'hamiltonian_map/laao_grid_msa.fasta' # grid output
output_pkl_path = 'hamiltonian_map/laao_grid_array.pkl' # pkl of grid
full_alignment = "MSA/filtered_20pct_pfam_full_PF01593_laao.fasta" # Family MSA
pixels = 500 # N^2 grid of sequences will be produced

def get_axis_values(input_model, training_set) -> int:
    loaded = load_model(model,compile=True)
    ds = tf.data.Dataset.from_generator(lambda: read_fasta_as_one_hot_encoded(training_set),tf.int8).batch(1000)
    _,_,zed = loaded.encoder.predict(ds)
    largest_value = abs(max(zed.min(),zed.max()))
    return int(np.ceil(largest_value))

square_max = get_axis_values(model,full_alignment)
axis_min = -square_max # x/y min
axis_max = square_max # x/y max


def get_key(val):
    for key, value in seq_code.items():
         if val == value:
             return key


def return_sequence(latent_output):
    seq = ''.join(get_key(x) for x in np.argsort(latent_output, axis=0)[-1,:])
    return seq


def make_grid_msa(model_path, output_path, resolution, batch_size=10000):
    trained_model = load_model(model_path)
    sampling_set = np.linspace(resolution[0], resolution[1], resolution[2])
    a = np.meshgrid(sampling_set, sampling_set)
    coord = np.vstack(np.array(a).transpose())
    with open(output_path, 'w') as fd:
        for batch_idx in range(0,coord.shape[0],batch_size):
            if batch_idx+batch_size > coord.shape[0]: #bigger than array
                z_input = coord[batch_idx:]
            else:
                z_input = coord[batch_idx:batch_idx+batch_size]
            latent_output = trained_model.decoder.predict(z_input)
            sequences = [return_sequence(seq_mat) for seq_mat in latent_output]
            for idx_seq, (x, y) in enumerate(z_input):
                fd.writelines('> ' + str(x) + ' ' + str(y) + '\n')
                fd.writelines(sequences[idx_seq] + '\n')
    fd.close()
    return coord


def get_hamiltonian(dca_fasta_in,model_grid_in, resolution, coords_for_pkl):
    mfdcamodel = dca(dca_fasta_in)
    mfdcamodel.mean_field()

    grid_hamiltonians, _ = mfdcamodel.compute_Hamiltonian(model_grid_in)

    grid_for_plotter = np.zeros((coords_for_pkl.shape[0],3))
    grid_for_plotter[:,:2] = coords_for_pkl
    grid_for_plotter[:,2] = grid_hamiltonians

    grid_hamiltonians = grid_hamiltonians.reshape(resolution[2],resolution[2])
    return grid_hamiltonians, grid_for_plotter


def plot_hamil_latent(hamil, title, resolution):

    # generate 2 2d grids for the x & y bounds
    a, b = np.meshgrid(np.linspace(resolution[0], resolution[1], resolution[2]),
                       np.linspace(resolution[0], resolution[1], resolution[2]))

    fig, ax = plt.subplots()
    c = ax.pcolormesh(b, a, hamil, cmap='jet')
    ax.set_title(title)
    fig.colorbar(c, ax=ax)
    plt.show()


figure_resolution = [axis_min, axis_max, pixels]


start = time.time()

# build coordinates

coordinates = make_grid_msa(model, output_path, figure_resolution)

# Run grid fasta through python mfdca
hamil_mat, grid_for_pkl = get_hamiltonian(full_alignment, output_path, figure_resolution, coordinates)

#save pickle file for plot_model.py
pickle.dump(grid_for_pkl,open(output_pkl_path,'wb'))

end = time.time()

print('Time Elapsed - '+str(end-start))
