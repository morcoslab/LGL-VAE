import subprocess
from collections import defaultdict

import numpy as np
from Bio import Seq, SeqIO

# load extant msa
msa_one = list(
    SeqIO.parse("MSA/fullheaders_globin_pfam_2022_filtered20pct.afa", "fasta")
)
msa_two = list(
    SeqIO.parse("MSA/top_fullheaders_desaturase_pfam_2022_filt20.afa", "fasta")
)

# get TaxID for sequences
headers_msa_one = list()
headers_msa_two = list()
for sequence in msa_one:
    try:
        headers_msa_one.append(sequence.description.split("OX=")[1].split()[0])
    except:
        headers_msa_one.append(
            "-1"
        )  # will never intersect with msa_two; no info in seq

for sequence in msa_two:
    try:
        headers_msa_two.append(sequence.description.split("OX=")[1].split()[0])
    except:
        headers_msa_two.append(
            "-2"
        )  # will never intersect with msa_one; no info in seq

# get list of taxid overlap (we can pair these successfully)
intersect_set = set(headers_msa_one).intersection(headers_msa_two)

# make key:value pair to link a header to a position in the msa
msa_one_dict = defaultdict(list)
msa_two_dict = defaultdict(list)
for idx, header in enumerate(headers_msa_one):
    if header in intersect_set:
        msa_one_dict[header].append(idx)
for idx, header in enumerate(headers_msa_two):
    if header in intersect_set:
        msa_two_dict[header].append(idx)


# append taxid to the front of the header, so the
# TreeDist package can compare taxid locations between the two trees
for label, sequence in zip(headers_msa_one, msa_one):
    sequence.id = label + " - " + sequence.id
for label, sequence in zip(headers_msa_two, msa_two):
    sequence.id = label + " - " + sequence.id

# generate msa!
for i in range(100):
    header_choice = np.random.choice(list(intersect_set), 640, replace=False)
    sampled_msa_one = [
        msa_one[np.random.choice(msa_one_dict[header])] for header in header_choice
    ]
    sampled_msa_two = [
        msa_two[np.random.choice(msa_two_dict[header])] for header in header_choice
    ]
    SeqIO.write(
        sampled_msa_one, "globin_msa/sampled_fasta_" + str(i + 1) + ".fasta", "fasta"
    )
    SeqIO.write(
        sampled_msa_two,
        "desaturase_msa/sampled_fasta_" + str(i + 1) + ".fasta",
        "fasta",
    )

# create trees!
for i in range(100):
    tree_one = "globin_trees/sampled_fasta_" + str(i + 1) + ".tree"
    tree_two = "desaturase_trees/sampled_fasta_" + str(i + 1) + ".tree"
    fasta_one = "globin_msa/sampled_fasta_" + str(i + 1) + ".fasta"
    fasta_two = "desaturase_msa/sampled_fasta_" + str(i + 1) + ".fasta"
    command_one = ["./FastTree", "-out", tree_one, fasta_one]
    command_two = ["./FastTree", "-out", tree_two, fasta_two]
    subprocess.run(command_one)
    subprocess.run(command_two)

# clustering distance is computed in R script.
