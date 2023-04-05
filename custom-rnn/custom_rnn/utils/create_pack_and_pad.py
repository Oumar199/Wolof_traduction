"""Create some functions to obtain better performances without using padding
"""
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from typing import Dict, Union
import numpy as np
import torch

def get_last_of_sequence(packed_sequence: PackedSequence, batch_first: bool = True):
    """Recuperate a tensor containing the last elements of the sequences

    Args:
        packed_sequence (PackedSequence): The packed sequence
        batch_first (bool, optional): Indicates if the batch are on the first position. Defaults to True.

    Returns:
        torch.Tensor: A tensor containing batch of sequences
    """
    
    
    # get padded sequences and sequence sizes
    pad_sequence, seq_sizes = pad_packed_sequence(packed_sequence, batch_first = batch_first)
    
    return pad_sequence[torch.arange(len(seq_sizes)), seq_sizes - 1]

class CustomPackedSequence:

    def __init__(self, sequences: Union[list, None] = None, data: Union[torch.Tensor, None] = None, batch_sizes: Union[list, None] = None, indices: Union[list, None] = None):
        """A class for transforming a list of sequences to a one sequenced tensor or to create a new custom packed sequence from data

        Args:
            sequences (list): List of sequences. Defaults to None.
            data (Union[torch.Tensor, None], optional): The data. Defaults to None.
            batch_sizes (Union[list, None], optional): The batch_sizes. Defaults to None.
            indices (Union[list, None], optional): _description_. Defaults to None.
        """
        
        if data != None and batch_sizes != None and indices != None:
            
            self.data = data
            
            self.indices = indices
        
        elif sequences != None:
            
            # d'abord verifions les longueurs des sequences ainsi que leurs indices
            length = []
            for i, seq in enumerate(sequences):
                
                length.append([len(seq), i])
            
            # trions les sequences
            length.sort(reverse=True)
            
            # stockons les sequences selon leur ordre de longueur
            ord_sequence = [sequences[j] for i, j in length]
            
            # maintenant nous allons stocker les sequences comme une seule sequence en prenant a chaque les longueurs
            one_sequence = []
            
            batch_sizes = []
            
            for l in range(length[0][0]):
                
                batch_size = 0
                
                for i, j in length:
                
                    try:
                        one_sequence.append(sequences[j][l])
                        batch_size += 1
                    except Exception:
                        break
                batch_sizes.append(batch_size)
            
            self.data = torch.from_numpy(np.stack(one_sequence))
            
            self.indices = np.array(length)[:, 1].tolist()
            
        else:
            
            raise ValueError("You must specify sequences or data, batch_sizes and indices !")
        
        self.batch_sizes = batch_sizes
    
    def to(self, device):
        
        self.data = self.data.to(device)
        
        return self
    
    def __print__(self):
        
        print({'data': self.data, 'indices': self.indices, 'batch_sizes': self.batch_sizes})
    
def pad_packed_sequence_(pack_sequence: CustomPackedSequence, batch_first: bool = False, return_last_on_sequence: bool = True):
    """A function which take a packed sequence and return the last element of each original sequence or return the padded original list of sequences
    as a tensor 

    Args:
        pack_sequence (CustomPackedSequence): The custom packed sequence object containing the packed sequences, the batch sizes and the indices
        batch_first (bool, optional): Indicate if we return a tensor with the batch dimension at first position. Defaults to False.
        return_last_on_sequence (bool, optional): Returns only the last elements of the sequences. Defaults to True.

    Returns:
        Union[torch.Tensor, tuple]: The outputs
    """
    
    # initialisons les sequences
    if batch_first:
        sequences = torch.zeros((len(pack_sequence.indices), len(pack_sequence.batch_sizes), pack_sequence.data.size(1)))
    else:
        sequences = torch.zeros((len(pack_sequence.batch_sizes), len(pack_sequence.indices), pack_sequence.data.size(1)))
        
    # nous allons iterer sur les tailles des batchs 
    n = 0
    seq_sizes = torch.zeros((len(pack_sequence.indices),))
    for i, batch in enumerate(pack_sequence.batch_sizes):
        
        for l in range(batch):
            
            seq_sizes[pack_sequence.indices[l]] += 1
            
            if batch_first:
                
                sequences[pack_sequence.indices[l]][i] = pack_sequence.data[n]
            
            else:
                
                sequences[i][pack_sequence.indices[l]] = pack_sequence.data[n]
                
            n+=1

    seq_sizes = seq_sizes.long()
    
    if return_last_on_sequence:
        
        if batch_first:    
                
            return sequences[torch.arange(sequences.size(0)), seq_sizes-1, :]
        
        else:
            
            return sequences[seq_sizes - 1, torch.arange(sequences.size(0)), :]
    
    return sequences, seq_sizes.tolist()


    
