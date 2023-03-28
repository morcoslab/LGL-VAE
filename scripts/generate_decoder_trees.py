import pickle
import subprocess

import numpy as np
import tensorflow as tf
from Bio import Seq, SeqIO
from dcavae.generators import read_fasta_as_one_hot_encoded
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture

aa_dict = {
    0: "G",
    1: "A",
    2: "V",
    3: "L",
    4: "I",
    5: "F",
    6: "P",
    7: "S",
    8: "T",
    9: "Y",
    10: "C",
    11: "M",
    12: "K",
    13: "R",
    14: "H",
    15: "W",
    16: "D",
    17: "E",
    18: "N",
    19: "Q",
    20: "U",
    21: "-",
    22: "O",
}


def create_cutoff_array(grid_array: np.array, hamiltonian_cutoff: float) -> np.array:
    """returns coordinates which should be disallowed based on cutoff"""
    return grid_array[
        np.arange(grid_array.shape[0])[grid_array[:, 2] > hamiltonian_cutoff], :2
    ]


def stack_awkward_arrays(list_of_arrays: list) -> np.array:
    final_array_length = sum([x.shape[0] for x in list_of_arrays])
    output_array = np.zeros((final_array_length, list_of_arrays[0].shape[1]))
    idx = 0
    for array in list_of_arrays:
        output_array[idx : idx + array.shape[0], :] = array
        idx += array.shape[0]
    return output_array


def generate_samples_with_hamiltonian_cutoff(
    mean: np.array,
    covariance: np.array,
    cutoff_array: np.array,
    distance_cutoff=0.01,
    num_samples=1000,
    batch_size=1000,
    verbose=False,
) -> np.array:
    """returns sample array to be converted into sequences by decoder"""
    output_list = list()
    mv_dist = np.random.multivariate_normal(mean, covariance, batch_size)
    init_distances = cdist(mv_dist, cutoff_array)
    output_list.append(
        mv_dist[init_distances.min(axis=1) > distance_cutoff]
    )  # this min dist works well
    while sum([len(x) for x in output_list]) < num_samples:
        if verbose == True:
            print("Num Samples = " + str(sum([len(x) for x in output_list])))
        mv_dist = np.random.multivariate_normal(mean, covariance, batch_size)
        add_distances = cdist(mv_dist, cutoff_array)
        output_list.append(mv_dist[add_distances.min(axis=1) > distance_cutoff])
    output_array = stack_awkward_arrays(output_list)
    return output_array[:num_samples]


def generate_random_decoder_output_hotone(
    vae_model: tf.keras.Model, z_inputs: np.array
) -> np.array:
    """takes in coordinates, outputs "truly" random sequences, not max prob"""
    decoded_sequences = vae_model.decoder.predict(z_inputs)
    index_array = np.arange(decoded_sequences.shape[1])
    numeric_msa = np.array(
        [[np.random.choice(index_array, p=x) for x in y.T] for y in decoded_sequences]
    )
    new_msa = np.zeros(decoded_sequences.shape)
    for seq in range(decoded_sequences.shape[0]):
        for pos in range(decoded_sequences.shape[2]):
            new_msa[seq, numeric_msa[seq, pos], pos] = 1
    return new_msa


def generate_sequences_from_coordinates(
    model: tf.keras.Model, z_coordinates: np.array, label: str
) -> list:
    """yields seqrecord array, which can be written to file with SeqIO.write"""
    generated_output = generate_random_decoder_output_hotone(model, z_coordinates)
    numeric_sequences = np.argsort(generated_output, axis=1)[:, -1, :]
    output_seq_list = []
    count = 0
    for sequence, coordinate in zip(numeric_sequences, z_coordinates):
        desc = " ".join([str(x) for x in coordinate])
        seq = Seq.Seq("".join([aa_dict[x] for x in sequence]))
        unique_id = "#" + str(count) + "-" + str(label)
        output_seq_list.append(
            SeqIO.SeqRecord(seq=seq, id=unique_id + " ; coord=", description=desc)
        )
        count += 1
    return output_seq_list


# load data and model
loaded_model = tf.keras.models.load_model("trained_models/globin")
fasta = "MSA/fullheaders_globin_pfam_2022_filtered20pct.afa"
ds = tf.data.Dataset.from_generator(
    lambda: read_fasta_as_one_hot_encoded(fasta), tf.int8
).batch(1000)
mu, logvar, z = loaded_model.encoder.predict(ds)

ham_map = pickle.load(open("hamiltonian_map/globin_grid_array.pkl", "rb"))
ham_grid = np.reshape(
    ham_map[:, 2], (int(np.sqrt(ham_map.shape[0])), int(np.sqrt(ham_map.shape[0])))
)
extent_dim = abs(ham_map[:, 0].min())


# create clusters
np.random.seed(10)
clusters = GaussianMixture(n_components=70).fit(mu)
labels = clusters.fit_predict(mu)

# create extant sampled trees
for z in range(100):
    listed_data = list(SeqIO.parse(fasta, "fasta"))
    output_list = []
    for i in range(len(clusters.means_)):
        if i not in bad_clusters:
            real_sequences_idx = [x for x in np.arange(len(listed_data))[labels == i]]
            random_sample = np.random.choice(real_sequences_idx, 10, replace=False)
            real_sequences = [listed_data[x] for x in random_sample]
            for seq in range(len(real_sequences)):
                real_sequences[seq].id = (
                    "#" + str(seq) + "-" + str(i) + " " + real_sequences[seq].id
                )
            output_list.extend(real_sequences)
    SeqIO.write(output_list, "subsampled_msa/globin_tree_" + str(z) + ".fasta", "fasta")

for z in range(100):
    msa = "subsampled_msa/globin_tree_" + str(z) + ".fasta"
    tree = "subsampled_trees/globin_tree_" + str(z) + ".tree"
    command = ["./FastTree", "-out", tree, msa]
    subprocess.run(command)

# create decoder generated trees
for z in range(100):
    output_list = []
    for i in range(len(clusters.means_)):
        if i not in bad_clusters:
            random_dist = generate_samples_with_hamiltonian_cutoff(
                clusters.means_[i],
                clusters.covariances_[i],
                cutoff_array,
                distance_cutoff=0.1,
                num_samples=10,
                verbose=False,
            )
            gen_seqs = generate_sequences_from_coordinates(
                loaded_model, random_dist, label=str(i)
            )
            output_list.extend(gen_seqs)
    SeqIO.write(
        output_list,
        "generated_msa/globin_seed10_70clusters_" + str(z + 1) + ".fasta",
        "fasta",
    )

for z in range(100):
    msa = "generated_msa/globin_seed10_70clusters_" + str(z + 1) + ".fasta"
    tree = "generated_trees/globin_seed10_70clusters_" + str(z + 1) + ".tree"
    command = ["./FastTree", "-out", tree, msa]
    subprocess.run(command)

# these trees are used in the R script.
