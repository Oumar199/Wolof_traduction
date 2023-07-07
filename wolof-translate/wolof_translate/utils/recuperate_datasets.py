from wolof_translate.utils.improvements.end_marks import add_end_mark
from wolof_translate.utils.sent_corrections import *

def recuperate_datasets(char_p: float, word_p: float, max_len: int, end_mark: int,
                        corpus_1: str = 'french', corpus_2: str = 'wolof', 
                        data_directory: str = 'data/extractions/new_data/'):

  # Let us recuperate the end_mark adding option
  if end_mark == 1:
    # Create augmentation to add on French sentences
    fr_augmentation_1 = TransformerSequences(nac.KeyboardAug(aug_char_p=char_p, aug_word_p=word_p,
                                                             aug_word_max = max_len),
                                          remove_mark_space, delete_guillemet_space, add_mark_space)

    fr_augmentation_2 = TransformerSequences(remove_mark_space, delete_guillemet_space, add_mark_space)
    
  else:
    
    if end_mark == 2:

      end_mark_fn = partial(add_end_mark, end_mark_to_remove = '!', replace = True)
    
    elif end_mark == 3:

      end_mark_fn = partial(add_end_mark)
    
    elif end_mark == 4:

      end_mark_fn = partial(add_end_mark, end_mark_to_remove = '!')
    
    else:  
        
        raise ValueError(f'No end mark number {end_mark}')

    # Create augmentation to add on French sentences
    fr_augmentation_1 = TransformerSequences(nac.KeyboardAug(aug_char_p=char_p, aug_word_p=word_p,
                                                             aug_word_max = max_len),
                                          remove_mark_space, delete_guillemet_space, add_mark_space, end_mark_fn)
    
    fr_augmentation_2 = TransformerSequences(remove_mark_space, delete_guillemet_space, add_mark_space, end_mark_fn)
    
  # Recuperate the train dataset
  train_dataset_aug = SentenceDataset(f"{data_directory}train_set.csv",
                                        tokenizer,
                                        truncation = False,
                                        cp1_transformer = fr_augmentation_1,
                                        cp2_transformer = fr_augmentation_2,
                                        corpus_1=corpus_1,
                                        corpus_2=corpus_2
                                        )

  # Recuperate the valid dataset
  valid_dataset = SentenceDataset(f"{data_directory}valid_set.csv",
                                        tokenizer,
                                        cp1_transformer = fr_augmentation_2,
                                        cp2_transformer = fr_augmentation_2,
                                        corpus_1=corpus_1,
                                        corpus_2=corpus_2,
                                        truncation = False)
  
  # Return the datasets
  return train_dataset_aug, valid_dataset
