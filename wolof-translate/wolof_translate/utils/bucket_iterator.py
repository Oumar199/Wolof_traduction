
import torch
from torch.utils.data import Sampler
from torch.nn.utils.rnn import pad_sequence

class SequenceLengthBatchSampler(Sampler):
    def __init__(self, dataset, boundaries, batch_sizes):
        self.dataset = dataset
        self.boundaries = boundaries
        self.batch_sizes = batch_sizes

    def __iter__(self):
        indices = list(range(len(self.dataset)))  # Get indices of the dataset
        sorted_indices = sorted(indices, key=lambda i: max(len(self.dataset[i][0]), len(self.dataset[i][1])))  # Sort indices based on sequence length
        self.batches = []

        # Group indices into batches of sequences with the same length
        for boundary in self.boundaries:
            batch = [i for i in sorted_indices if len(self.dataset[i][0]) <= boundary]  # Filter indices based on length boundary
            self.batches.append(batch)
            sorted_indices = [i for i in sorted_indices if i not in batch]  # Remove processed indices

        # Add remaining indices to the last batch
        self.batches.append(sorted_indices)

        # Yield batches with the corresponding batch sizes
        for batch_indices, batch_size in zip(self.batches, self.batch_sizes):
            for i in range(0, len(batch_indices), batch_size):
                yield batch_indices[i:i + batch_size]

    def __len__(self):
        return sum(len(batch) // batch_size + 1 for batch, batch_size in zip(self.batches, self.batch_sizes))

def collate_fn(batch):
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
