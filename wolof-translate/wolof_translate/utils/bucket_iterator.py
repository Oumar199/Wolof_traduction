
import torch
import numpy as np
from torch.utils.data import Sampler
from torch.nn.utils.rnn import pad_sequence
from math import ceil

class SequenceLengthBatchSampler(Sampler):
    def __init__(self, dataset, boundaries, batch_sizes):
        self.dataset = dataset
        self.boundaries = boundaries
        self.batch_sizes = batch_sizes
        self.data_info = {}

        # Initialize dictionary with indices and element lengths
        for i, data in enumerate(dataset):
            length = max(len(data[0]), len(data[2]))
            self.data_info[i] = {"index": i, "length": length}

        self.calculate_length()

    def calculate_length(self):
        self.batches = []

        # Sort indices based on element length
        sorted_indices = sorted(self.data_info.keys(), key=lambda i: self.data_info[i]["length"])

        # Group indices into batches of sequences with the same length
        for boundary in self.boundaries:
            batch = [i for i in sorted_indices if self.data_info[i]["length"] <= boundary]  # Filter indices based on length boundary
            self.batches.append(batch)
            sorted_indices = [i for i in sorted_indices if i not in batch]  # Remove processed indices

        # Add remaining indices to the last batch
        self.batches.append(sorted_indices)

        # Calculate the total length of the data loader
        self.length = sum(ceil(len(batch) / batch_size) for batch, batch_size in zip(self.batches, self.batch_sizes))

    def __iter__(self):
        indices = list(self.data_info.keys())  # Get indices from the data_info dictionary
        np.random.shuffle(indices)  # Shuffle the indices

        # Yield batches with the corresponding batch sizes
        for batch_indices, batch_size in zip(self.batches, self.batch_sizes):
            num_batches = len(batch_indices) // batch_size

            for i in range(num_batches):
                # Recuperate the current bucket
                current_bucket = batch_indices[i * batch_size: (i + 1) * batch_size]

                # Shuffle the current bucket
                np.random.shuffle(current_bucket)

                # Yield the current bucket
                yield [self.data_info[i]["index"] for i in current_bucket]

            remaining_indices = len(batch_indices) % batch_size
            if remaining_indices > 0:
                # Recuperate the current bucket
                current_bucket = batch_indices[-remaining_indices:]

                # Shuffle the current bucket
                np.random.shuffle(current_bucket)

                # Yield the current bucket
                yield [self.data_info[i]["index"] for i in batch_indices[-remaining_indices:]]

    def __len__(self):
        return self.length


class BucketSampler(Sampler):
    def __init__(self, dataset, batch_size, sort_key=lambda x: max(len(x[0]), len(x[1]))):
        self.dataset = dataset
        self.batch_size = batch_size
        self.sort_key = sort_key

    def __iter__(self):
        indices = np.argsort([self.sort_key(self.dataset[i]) for i in range(len(self.dataset))])
        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        if self.batch_size > 1:
            np.random.shuffle(batches)
        for batch in batches:
            yield batch.tolist()

    def __len__(self):
        return ceil(len(self.dataset) / self.batch_size)


def collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence
    # Separate the input sequences, target sequences, and attention masks
    input_seqs, input_masks, target_seqs, target_masks = zip(*batch)

    # Pad the input sequences to have the same length
    padded_input_seqs = pad_sequence(input_seqs, batch_first=True)

    # Pad the target sequences to have the same length
    padded_target_seqs = pad_sequence(target_seqs, batch_first=True)

    # Pad the input masks to have the same length
    padded_input_masks = pad_sequence(input_masks, batch_first=True)

    # Pad the labels masks to have the same length
    padded_target_masks = pad_sequence(target_masks, batch_first=True)

    return padded_input_seqs, padded_input_masks, padded_target_seqs, padded_target_masks
