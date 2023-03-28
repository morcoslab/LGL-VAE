import numpy as np
from Bio import SeqIO


seq_code = {'G': 0,  # Amino acids including selenocysteine and gap character
            'A': 1,
            'V': 2,
            'L': 3,
            'I': 4,
            'F': 5,
            'P': 6,
            'S': 7,
            'T': 8,
            'Y': 9,
            'C': 10,
            'M': 11,
            'K': 12,
            'R': 13,
            'H': 14,
            'W': 15,
            'D': 16,
            'E': 17,
            'N': 18,
            'Q': 19,
            'U': 20,
            '-': 21,
            'O': 22,
            'X': range(0, 23),
            'B': [16, 18],
            'Z': [17, 19],
            'J': [3, 4]}


def get_sequence_length(file_path):
    parser = SeqIO.parse(file_path, 'fasta')
    record = parser.__next__()
    return len(record)


def get_fasta_file_dimensions(file_path):
    """ Loop through file once to get the number of valid FASTA entries"""

    parser = SeqIO.parse(file_path, 'fasta')
    num_sequences = 0
    length_sequence = 0

    try:
        first_record = next(parser)
        length_sequence = len(first_record.seq)
        num_sequences = 1

        for _ in parser:
            num_sequences += 1

    except StopIteration:
        pass

    return num_sequences, length_sequence


# Load data
def read_fasta_as_one_hot_encoded(file_path):
    for record in SeqIO.parse(file_path, 'fasta'):
        seq_len = len(record.seq)
        seq_one_hot = np.zeros((23, seq_len))
        for idx2, aa in enumerate(record.seq[:seq_len]):
            seq_one_hot[seq_code[aa.upper()], idx2] = 1

        yield seq_one_hot.flatten()
