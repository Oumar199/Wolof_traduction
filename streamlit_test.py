from transformers import T5ForConditionalGeneration, T5TokenizerFast
from torch.utils.data import DataLoader
from tqdm import tqdm
import streamlit as st
import pandas as pd
import torch
import os


# Let us define the main page
st.markdown("Translation page 🔠")

# Dropdown for the translation type
translation_type = st.sidebar.selectbox("Translation Type", options=["French ➡️ Wolof", "Wolof ➡️ French"])

# define a dictionary of versions
models = {
    "Version ✌️": {
        "French ➡️ Wolof": {
            "checkpoints": "wolof-translate/wolof_translate/checkpoints/t5_small_custom_train_results_fw_v4",
            "tokenizer": "wolof-translate/wolof_translate/tokenizers/t5_tokenizers/tokenizer_v4.json",
            "max_len": None
        }
    },
    "Version ☝️": {
        "French ➡️ Wolof": {
            "checkpoints": "wolof-translate/wolof_translate/checkpoints/t5_small_custom_train_results_fw_v3",
            "tokenizer": "wolof-translate/wolof_translate/tokenizers/t5_tokenizers/tokenizer_v3.json",
            "max_len": 51
            }
    }
}

# Dropdown for the model version
version = st.sidebar.selectbox("Model version", options=["Version ☝️", "Version ✌️"])

# Recuperate the number of sentences to provide
number = st.sidebar.number_input("Give the number of sentences that you want to provide", min_value = 1,
          max_value = 100)

# Recuperate the number of sentences to provide
temperature = st.sidebar.slider("How randomly need you the translated sentences to be from 0% to 100%", min_value = 0,
          max_value = 100)


# make the process
try:
    # recuperate checkpoints
    checkpoints = torch.load(os.path.join(models[version][translation_type]['checkpoints'], "best_checkpoints.pth"))

    # recuperate the tokenizer
    tokenizer_file = models[version][translation_type]['tokenizer']
    
    # recuperate the max length
    max_len = models[version][translation_type]['max_len']
    
    # let us get the best model
    @st.cache_resource
    def get_model():
        
        # initialize the tokenizer
        tokenizer = T5TokenizerFast(tokenizer_file=tokenizer_file)
        
        # initialize the model
        model_name = 't5-small'
        
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # resize the token embeddings
        model.resize_token_embeddings(len(tokenizer))
        
        model.load_state_dict(checkpoints['model_state_dict'])
        
        
        return model, tokenizer

    model, tokenizer = get_model()

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
        
        sentence = st.session_state[f"sentence{i}"] + tokenizer.eos_token
        
        if not sentence == "":
        
            # Let us encode the sentences
            encoding = tokenizer([sentence], return_tensors='pt', max_length=max_len, padding='max_length', truncation=True)
            
            # Let us recuperate the input ids
            input_ids = encoding.input_ids
            
            # Let us recuperate the mask
            mask = encoding.attention_mask
            
            # Let us recuperate the pad token id
            pad_token_id = tokenizer.pad_token_id
            
            # perform prediction
            predictions = model.generate(input_ids, do_sample = False, top_k = 50, max_length = max_len, top_p = 0.90,
                                            temperature = temperature/100, num_return_sequences = 0, attention_mask = mask, pad_token_id = pad_token_id)
            
            # decode the predictions
            predicted_sentence = tokenizer.batch_decode(predictions, skip_special_tokens = True)
            
            # provide the prediction
            right.write(f"{i+1}. {predicted_sentence[0]}")
            
        else:
            
            # provide the prediction
            right.write(f"{i+1}. ")
    
except Exception as e:
    
    st.warning("The chosen model is not available yet !", icon = "⚠️")
    
    # st.write(e)
    


