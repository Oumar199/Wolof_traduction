""" This module contains a function which split the data. It will consider adding the validation set
"""
from sklearn.model_selection import train_test_split
import pandas as pd
import os

def split_data(random_state: int = 50, data_directory: str = "data/extractions/new_data"):
  """Split data between train, validation and test sets

  Args:
    random_state (int): the seed of the splitting generator. Defaults to 50
  """
  # load the corpora and split into train and test sets
  corpora = pd.read_csv(os.path.join(data_directory, "sentences.csv"))

  train_set, test_set = train_test_split(corpora, test_size=0.1, random_state=random_state)

  # let us save the final training set when performing

  train_set, valid_set = train_test_split(train_set, test_size=0.1, random_state=random_state)

  train_set.to_csv(os.path.join(data_directory, "final_train_set.csv"), index=False)

  # let us save the sets
  train_set.to_csv(os.path.join(data_directory, "train_set.csv"), index=False)

  valid_set.to_csv(os.path.join(data_directory, "valid_set.csv"), index=False)

  test_set.to_csv(os.path.join(data_directory, "test_set.csv"), index=False)
