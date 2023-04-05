"""
Generate names
-------------------------
We generate names after some transformations:
- Normalize letters on names
- Make one hot encoding on random chosen name
- Change a category to number
"""
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


class NameDataset(Dataset):
    def __init__(self, path: str):
        self.path = path
        self.names = []
        self.categories = {}
        self.length = 0
        for file in glob.glob(os.path.join(self.path, "*.txt")):
            category = os.path.splitext(os.path.basename(file))[0]
            lines = [
                self.normalize(line)
                for line in io.open(file, encoding="utf-8").read().strip().split("\n")
            ]
            self.length += len(lines)
            self.categories[category] = lines
        self.classes = list(self.categories.keys())

    def normalize(self, line):
        return "".join(
            [
                char
                for char in unicodedata.normalize("NFD", line)
                if unicodedata.category(char) != "Mn" and char in LETTERS
            ]
        )

    def __getitem__(self, index):
        category = random.choice(self.classes)
        name = random.choice(self.categories[category])
        encoded_name = torch.zeros(len(name), N_LETTERS)
        for i in range(encoded_name.size(0)):
            encoded_letter = self.one_hot_encoding(name[i])
            encoded_name[i] = encoded_letter
        category_class = self.classes.index(category)

        return name, category, encoded_name, torch.tensor(category_class).unsqueeze(0)

    def one_hot_encoding(self, letter):
        encoded_letter = torch.zeros(N_LETTERS)
        letter_position = LETTERS.find(letter)
        encoded_letter[letter_position] = 1
        return encoded_letter

    def __len__(self):
        return self.length


"""Let's create a second custom dataset which will give us batches in place of one single random batch at each iteration
"""


class NameDataset2(Dataset):
    def __init__(self, path: str):
        self.path = path
        self.names = []
        self.classes = []
        self.categories = []
        self.max_len = 0
        self.length = 0
        for file in glob.glob(os.path.join(self.path, "*.txt")):
            category = os.path.splitext(os.path.basename(file))[0]
            lines = []
            for line in io.open(file, encoding="utf-8").read().strip().split("\n"):
                if len(line) > self.max_len:
                    self.max_len = len(line)

                lines.append(self.normalize(line))

            self.length += len(lines)
            self.names.extend(lines)
            self.categories.extend([category] * len(lines))
            self.classes.append(category)
            self.weights = (
                1 / pd.Series(self.categories).value_counts(normalize=True)
            ).to_list()

    def normalize(self, line):
        return "".join(
            [
                char
                for char in unicodedata.normalize("NFD", line)
                if unicodedata.category(char) != "Mn" and char in LETTERS
            ]
        )

    def __getitem__(self, index):
        category = self.categories[index]
        name = self.names[index]
        encoded_name = torch.zeros(len(name), N_LETTERS)
        for i in range(encoded_name.size(0)):
            encoded_letter = self.one_hot_encoding(name[i])
            encoded_name[i] = encoded_letter
        category_class = self.classes.index(category)
        padding = self.max_len - encoded_name.shape[0]
        return (
            F.pad(encoded_name, (0, 0, padding, 0)),
            torch.tensor(category_class).unsqueeze(0),
            name,
            category,
        )

    def one_hot_encoding(self, letter):
        encoded_letter = torch.zeros(N_LETTERS)
        letter_position = LETTERS.find(letter)
        encoded_letter[letter_position] = 1
        return encoded_letter

    def __len__(self):
        return self.length
