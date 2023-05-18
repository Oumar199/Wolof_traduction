
from transformers import AutoModelForSeq2SeqLM, T5TokenizerFast
from torch.utils.data import DataLoader
from tqdm import tqdm
import streamlit as st
import pandas as pd
import torch


# Let us define the main page
st.markdown("Translation page 🔠")

# Dropdown
translation_type = st.sidebar.selectbox("Translation Type", options=["French ➡️ Wolof", "Wolof ➡️ French"])

# Recuperate the number of sentences to provide
number = st.sidebar.number_input("Give the number of sentences that you want to provide", min_value = 1,
          max_value = 100)

# Recuperate the number of sentences to provide
temperature = st.sidebar.slider("How randomly need you the translated sentences to be from 0% to 100%", min_value = 0,
          max_value = 100)


# recuperate the tokenizer from a json file
tokenizer = T5TokenizerFast(tokenizer_file=f"wolof-translate/wolof_translate/tokenizers/t5_tokenizers/tokenizer_v3.json")

# let us get the best model
@st.cache_resource
def get_model():
    
    model = AutoModelForSeq2SeqLM.from_pretrained('data/checkpoints/t5_results_fw_v3/checkpoint-151698/')

    return model

model = get_model()

# set the model to eval mode
_ = model.eval()

# Add a title
st.header("Translate French sentences onto Wolof 👌")


# Recuperate two columns
left, right = st.columns(2)

# recuperate sentences
left.subheader('Give me some sentences in French: ')

for i in range(number):
    
    left.text_input(f"- Sentence number {i + 1}", key = f"sentence{i}")

# run model inference on all test data
original_translations, predicted_translations, original_texts, scores = [], [], [], {}

# print a sentence recuperated from the session
right.subheader("Translation to Wolof:")

for i in range(number):
    
    sentence = st.session_state[f"sentence{i}"]
    
    if not sentence == "":
    
        # Let us encode the sentences
        encoding = tokenizer([sentence], return_tensors='pt', max_length=51, padding='max_length', truncation=True)
        
        # Let us recuperate the input ids
        input_ids = encoding.input_ids
        
        # Let us recuperate the mask
        mask = encoding.attention_mask
        
        # Let us recuperate the pad token id
        pad_token_id = tokenizer.pad_token_id
        
        # perform prediction
        predictions = model.generate(input_ids, do_sample = False, top_k = 50, max_length = 51, top_p = 0.90,
                                        temperature = temperature/100, num_return_sequences = 0, attention_mask = mask, pad_token_id = pad_token_id)
        
        # decode the predictions
        predicted_sentence = tokenizer.batch_decode(predictions, skip_special_tokens = True)
        
        # provide the prediction
        right.write(f"{i+1}. {predicted_sentence[0]}")
        
    else:
        
        # provide the prediction
        right.write(f"{i+1}. ")
        
    
