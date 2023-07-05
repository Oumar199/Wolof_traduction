
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

class SameLengthBatchSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size

    def __iter__(self):
        indices = list(range(len(self.data_source)))
        indices.sort(key=lambda i: len(self.data_source[i][0]), reverse=True)
        for i in range(0, len(indices), self.batch_size):
            print('here here')
            yield indices[i:i + self.batch_size]

    def __len__(self):
        return len(self.data_source) // self.batch_size

# class SameLengthBatchSampler(torch.utils.data.Sampler):
#     def __init__(self, data_source, boundaries, batch_sizes):
#         self.data_source = data_source
#         self.boundaries = boundaries
#         self.batch_sizes = batch_sizes

#     def __iter__(self):
#         indices = list(range(len(self.data_source)))
#         indices.sort(key=lambda i: len(self.data_source[i][0]), reverse=True)
#         batches = []
#         current_batch = []
#         current_bucket_index = 0
#         for i in indices:
#             length = len(self.data_source[i][0])
#             if length > self.boundaries[current_bucket_index]:
#                 if current_bucket_index < len(self.batch_sizes):
#                     batches.extend([current_batch] * self.batch_sizes[current_bucket_index])
#                 else:
#                     batches.append(current_batch)
#                 current_batch = [i]
#                 current_bucket_index += 1
#             else:
#                 current_batch.append(i)
#         if current_batch:
#             if current_bucket_index < len(self.batch_sizes):
#                 batches.extend([current_batch] * self.batch_sizes[current_bucket_index])
#             else:
#                 batches.append(current_batch)
#         for batch in batches:
#             print(batch)

#     def __len__(self):
#         num_batches = 0
#         for i in range(len(self.batch_sizes)):
#             if i < len(self.boundaries):
#                 num_batches += len(self.data_source) // self.batch_sizes[i]
#             else:
#                 num_batches += len(self.data_source) // self.batch_sizes[-1]
#         return num_batches

def collate_fn(batch):
    # Sort the batch based on input sequence length (descending order)
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    
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
