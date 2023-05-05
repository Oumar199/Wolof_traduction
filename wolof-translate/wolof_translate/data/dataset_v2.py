from wolof_translate.utils.sent_transformers import TransformerSequences
from transformers import PreTrainedTokenizerFast
from torch.utils.data import Dataset
from typing import *
import pandas as pd
import torch
import re

class T5SentenceDataset(Dataset):

    def __init__(
        self,
        data_path: str, 
        tokenizer: PreTrainedTokenizerFast,
        corpus_1: str = "french",
        corpus_2: str = "wolof",
        max_len: int = 51,
        truncation: bool = False,
        file_sep: str = ",",
        cp1_transformer: Union[TransformerSequences, None] = None,
        cp2_transformer: Union[TransformerSequences, None] = None,
        **kwargs):
        
        # let us recuperate the data frame
        self.__sentences = pd.read_csv(data_path, sep=file_sep, **kwargs)
        
        # let us recuperate the tokenizer
        self.tokenizer = tokenizer
        
        # recuperate the first corpus' sentences
        self.__sentences_1 = self.__sentences[corpus_1].to_list()
        
        # recuperate the second corpus' sentences
        self.__sentences_2 = self.__sentences[corpus_2].to_list()
        
        # recuperate the length
        self.__length = len(self.__sentences_1)
        
        # let us recuperate the max len
        self.max_len = max_len
        
        # let us recuperate the truncation argument
        self.truncation = truncation
        
        # let us initialize the transformer
        self.cp1_transformer = cp1_transformer
        
        self.cp2_transformer = cp2_transformer
        
    def __getitem__(self, index):
        """Recuperate ids and attention masks of sentences at index

        Args:
            index (int): The index of the sentences to recuperate

        Returns:
            tuple: The `sentence to translate' ids`, `the attention mask of the sentence to translate`
            `the labels' ids`
        """
        sentence_1 = self.__sentences_1[index]
        
        sentence_2 = self.__sentences_2[index]
        
        # apply transformers if necessary
        if not self.cp1_transformer is None:
            
            sentence_1 = self.cp1_transformer(sentence_1) 
        
        if not self.cp2_transformer is None:
            
            sentence_2 = self.cp2_transformer(sentence_2)
        
        # let us encode the sentences (we provide the second sentence as labels to the tokenizer)
        data = self.tokenizer(
            sentence_1,
            truncation=self.truncation,
            max_length=self.max_len, 
            padding='max_length', 
            return_tensors="pt",
            text_target=sentence_2)
            
        
        return data.input_ids.squeeze(0), data.attention_mask.squeeze(0), data.labels.squeeze(0)
        
    def __len__(self):
        
        return self.__length
    
    def decode(self, labels: torch.Tensor):
        
        if labels.ndim < 2:
            
            labels = labels.unsqueeze(0)

        sentences = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        return sentences
