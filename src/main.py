# --- Downloading utilities ---
import wget  # Used to programmatically download files (e.g., datasets or models)

# --- Hugging Face Transformers (for DPR and language models) ---
from transformers import (
    DPRContextEncoder, 
    DPRContextEncoderTokenizer,
    DPRQuestionEncoder, 
    DPRQuestionEncoderTokenizer,
    AutoTokenizer, 
    AutoModelForCausalLM
)

# --- Core ML and computation libraries ---
import torch      # PyTorch for tensor computation and model inference
import numpy as np  # Numpy for numerical operations
import random       # Randomness control for reproducibility

# --- Visualization tools ---
import matplotlib.pyplot as plt  # Plotting library
from mpl_toolkits.mplot3d import Axes3D  # Enables 3D plotting with matplotlib
from sklearn.manifold import TSNE        # Dimensionality reduction for visualization

# --- Warning suppression (optional but keeps notebooks tidy) NOT FOR PRODUCTION CODE ---
def warn(*args, **kwargs):
    """Custom warning handler to suppress unwanted warnings."""
    pass

import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

import sys, os
import contextlib # Provides utilities for working with context managers (e.g., redirecting stdout/stderr, closing resources)

filename = 'companyPolicies.txt'
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/6JDbUb_L3egv_eOkouY71A.txt'

# Use wget to download the file
wget.download(url, out=filename)
print('file downloaded')

def read_and_split_text(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    # Split the text into paragraphs (simple split by newline characters)
    paragraphs = text.split('\n')
    # Filter out any empty paragraphs or undesired entries
    paragraphs = [para.strip() for para in paragraphs if len(para.strip()) > 0]
    return paragraphs

# Read the text file and split it into paragraphs
paragraphs = read_and_split_text('companyPolicies.txt')
paragraphs[0:10]

'''Initialize the tokenizer that converts raw text passages into input IDs and attention masks
compatible with the DPR context encoder model. DPRContextEncoderTokenizer object is identical to BertTokenizer and runs end-to-end tokenization including punctuation splitting and wordpiece'''

#  # Option 1: simplest – just load it normally
# context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
#     'facebook/dpr-ctx_encoder-single-nq-base'
# )
# # Option 2: if you want to suppress output (similar to %%capture in Jupyter)
with open(os.devnull, "w") as f, \
     contextlib.redirect_stdout(f), \
     contextlib.redirect_stderr(f):
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
        "facebook/dpr-ctx_encoder-single-nq-base"
    )
print("Tokenizer loaded successfully!")


''' TOKENIZATION:
    Example of tokeniser using simple sample that can relate back to BERT:
'''

# text = [("How are you?", "I am fine."), ("What's up?", "Not much.")]
# print(text)

# tokens_info=context_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256)
# print("Tokens Info: \n", tokens_info)

'''token_type_ids: These are segment embeddings used to distinguish different sentences or segments within the input. This is particularly useful in tasks that involve multiple types of input, such as question answering, where questions and context may need to be differentiated.

attention_mask: The attention mask indicates which tokens should be attended to by the model. It has a value of 1 for actual tokens in the input sentences and 0 for padding tokens, ensuring that the model focuses only on meaningful data.

input_ids: These represent the indices of tokens in the tokenizer's vocabulary. A tokenizer has a vocabulary — basically a big list (array) of tokens, each with a number. When you feed text through a tokenizer, it turns words/subwords into their numeric positions in this vocabulary list. 

For DPRContextEncoderTokenizer, the vocab usually comes from BERT, so the file is:
vocab.txt
When you run...
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
    "facebook/dpr-ctx_encoder-single-nq-base"
)
...Hugging Face downloads a folder to your machine at a path like:
~/.cache/huggingface/hub/models--facebook--dpr-ctx_encoder-single-nq-base/snapshots/<hash>/
Inside that folder you will find:
- vocab.txt
- tokenizer_config.json
- special_tokens_map.json
Each line is a token, and the line number is the index.
The inout_ids/indices are literally the index of each token inside vocab.txt.

To translate these indices back into readable tokens, you can use the method convert_ids_to_tokens provided by the tokenizer. See below to use this method:
'''

# for s in tokens_info['input_ids']:
#    print(context_tokenizer.convert_ids_to_tokens(s))

