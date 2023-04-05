
import io
import os
import glob
import string
import torch
import random
import numpy as np
import unicodedata
import pandas as pd
from torch import nn
from torch.utils.data import Dataset
from torch.nn import functional as F

LETTERS = string.ascii_letters + ",.-_;"
N_LETTERS = len(LETTERS)

class HalfNameDataset(Dataset):
    def __init__(self, path: str):
        
        # recuperate the path containing the text files
        self.path = path
        
        # the first names' halves in letter
        self.first_halves = []
        
        # the second names' halves in letter
        self.second_halves = []
        
        for file in glob.glob(os.path.join(self.path, "*.txt")):
            
            # we normalize the letters in the text files and recuperate them
            for line in io.open(file, encoding = "utf-8").read().strip().split("\n"):
                
                # normalize the name
                norm_name = self.normalize(line)
                
                # recuperate the length of the normalized name
                name_len = len(norm_name)
                
                # recuperate the length of the first half
                first_half_len = name_len // 2
                
                # recuperate the first half
                self.first_halves.append(norm_name[:first_half_len])
                
                # recuperate the second half
                self.second_halves.append(norm_name[first_half_len:])
                
        # recuperate the length of the dataset
        self.length = len(self.first_halves)
    
    def normalize(self, line):
        return "".join([char for char in unicodedata.normalize("NFD", line) if unicodedata.category(char) != "Mn" and char in LETTERS])
    
    def __getitem__(self, index):
        
        # recuperate the first half at the given index 
        first_half = self.first_halves[index]
        
        # recuperate the second half at the given index
        second_half = self.second_halves[index]
        
        # recuperate the positions
        first_half_positions = [LETTERS.find(letter) for letter in first_half if letter in LETTERS]
        
        second_half_positions = [LETTERS.find(letter) for letter in second_half if letter in LETTERS]
        
        return torch.tensor(first_half_positions), torch.tensor(second_half_positions)
    
    def traduce(self, positions):
        
        return "".join([LETTERS[position] for position in positions])
    
    def encode(self, name: str, normalize: bool = False):
        """A new class method which take a name (characters) and converts each character to a position in the corpus

        Args:
            name (str): The name
        """
        if normalize:
            
            # we normalize the name if necessary
            name = self.normalize(name)
            
        # recuperate the positions of the letter
        positions = [LETTERS.find(letter) for letter in name if letter in LETTERS]
        
        return positions
    
    def __len__(self):
        return self.length

