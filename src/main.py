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
from utils.visualisation import tsne_plot

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


# ''' TOKENIZATION:
#     Example of tokeniser using simple sample that can relate back to BERT:
# '''

# # text = [("How are you?", "I am fine."), ("What's up?", "Not much.")]
# # print(text)

# # tokens_info=context_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256)
# # print("Tokens Info: \n", tokens_info)

# '''token_type_ids: These are segment embeddings used to distinguish different sentences or segments within the input. This is particularly useful in tasks that involve multiple types of input, such as question answering, where questions and context may need to be differentiated.

# attention_mask: The attention mask indicates which tokens should be attended to by the model. It has a value of 1 for actual tokens in the input sentences and 0 for padding tokens, ensuring that the model focuses only on meaningful data.

# input_ids: These represent the indices of tokens in the tokenizer's vocabulary. A tokenizer has a vocabulary — basically a big list (array) of tokens, each with a number. When you feed text through a tokenizer, it turns words/subwords into their numeric positions in this vocabulary list. 

# For DPRContextEncoderTokenizer, the vocab usually comes from BERT, so the file is:
# vocab.txt
# When you run...
# context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
#     "facebook/dpr-ctx_encoder-single-nq-base"
# )
# ...Hugging Face downloads a folder to your machine at a path like:
# ~/.cache/huggingface/hub/models--facebook--dpr-ctx_encoder-single-nq-base/snapshots/<hash>/
# Inside that folder you will find:
# - vocab.txt
# - tokenizer_config.json
# - special_tokens_map.json
# Each line is a token, and the line number is the index.
# The inout_ids/indices are literally the index of each token inside vocab.txt.

# To translate these indices back into readable tokens, you can use the method convert_ids_to_tokens provided by the tokenizer. See below to use this method:
# '''

# # for s in tokens_info['input_ids']:
# #    print(context_tokenizer.convert_ids_to_tokens(s))


# ''' ENCODING
# The tokenized texts are then fed into the context_encoder. This model processes the inputs and produces a pooled output for each, effectively compressing the information of an entire text into a single, dense vector embedding that represents the semantic essence of the text.

# DPR (Dense Passage Retriever) models, including the ```DPRContextEncoder```, are based on the BERT architecture but specialize in dense passage retrieval. They differ from BERT in their training, which focuses on contrastive learning for retrieving relevant passages, while BERT is more general-purpose, handling various NLP tasks.
# '''

# context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

# '''
# The context_tokenizer and context_encoder work together to process text data, transforming paragraphs into contextual embeddings suitable for further NLP tasks. Here's how these components are applied to the first 20 paragraphs from a list:

# The context_tokenizer takes the first 20 paragraphs and converts each into a sequence of token IDs, formatted specifically as input to a PyTorch model. This process includes:
# Padding: To ensure uniformity, shorter text sequences are padded with zeros to reach the specified maximum length of 256 tokens.
# Truncation: Longer texts are cut off at 256 tokens to maintain consistency across all inputs.
# The tokenized data is then passed to the context_encoder, which processes these token sequences to produce contextual embeddings. Each output embedding vector from the encoder represents the semantic content of its corresponding paragraph, encapsulating key informational and contextual nuances.
# The encoder outputs a PyTorch tensor where each row corresponds to a different paragraph's embedding. The shape of this tensor, determined by the number of paragraphs processed and the embedding dimensions, reflects the detailed, contextualized representation of each paragraph's content.'''

# # shuffle samples so that the samples are not ordered based on the category they belong to
# random.seed(42)
# random.shuffle(paragraphs)

# # tokenize first 20 paragraphs with padding, truncation, and max_length=256
# tokens=context_tokenizer( paragraphs[:20], return_tensors='pt', padding=True, truncation=True, max_length=256) 
# tokens

# #  pass tokenized inputs into DPRContextEncoder using argument unpacking (**tokens)
# outputs=context_encoder(**tokens)
# print("Encoded Output: /n", outputs.pooler_output)


# ''' t-SNE (t-Distributed Stochastic Neighbor Embedding)
# t-SNE is an effective method for visualizing high-dimensional data, making it particularly useful for analyzing outputs from DPRContextEncoder models. The DPRContextEncoder encodes passages into dense vectors that capture their semantic meanings within a high-dimensional (e.g 768dim) space. Applying t-SNE to these dense vectors allows you to reduce their dimensionality to two or three dimensions. This reduction creates a visual representation that preserves the relationships between passages, enabling you to explore clusters of similar passages and discern patterns that might otherwise remain hidden in the high-dimensional space. The resulting plots provide insights into how the model differentiates between different types of passages and reveal the inherent structure within the encoded data.'''

# # tsne_plot(outputs.pooler_output.detach().numpy())

# # # Samples 16 and 12 are closer to each other on the graph produced above
# # print("sample 16:", paragraphs[16])
# # print("sample 12:", paragraphs[12])
# '''Both samples discuss diversity. Rather than relying solely on visual inspection, distances between embeddings are employed to determine the relevance of retrieved documents or passages. This involves comparing the query’s embedding with the embeddings of candidate documents, enabling a precise and objective measure of relevance.'''

# ''' AGGREGATION
# All individual embeddings generated from the texts are then aggregated into a single NumPy array. This aggregation is essential for subsequent processing steps, such as indexing, which facilitates efficient similarity searches.

# This methodological approach efficiently transforms paragraphs into a form that retains crucial semantic information in a compact vector format, making it ideal for the retrieval tasks necessary in this lab.
# '''

# # compile a list containing each sample, where each sample has specific dimensions
# embeddings=[]
# for text in paragraphs[0:5]:
#     inputs = context_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256)
#     outputs = context_encoder(**inputs)
#     embeddings.append(outputs.pooler_output)
#     print("number of samples: ", len(embeddings))
#     print(" samples shape: ",  outputs.pooler_output.shape)

# # → concatenate tensors into a single 2D tensor, shape (num_samples, hidden_size)
# # → remove computation graph
# # → convert result into a NumPy array
# '''
# Every tensor produced by a model tracks gradients; 
# A gradient is a derivative. A derivative measures how fast something changes when you change something else. e.g. “If I change this weight a tiny bit, how much will the error change?”; Each operation is recorded, so PyTorch can later compute gradients during backpropagation; 
# Here is just encoding text or generating embeddings: this case is NOT training, doed NOT want gradients tracking gradients wastes memory and compute; don't want gradients for inference, because only computing embeddings: saves memory, prevents accidental backprop, makes it safe to convert to numpy; NumPy knows nothing about PyTorch’s computation graph; A graph-tracked tensor cannot be exported.
# # '''
# torch.cat(embeddings).detach().numpy().shape


''' CONSOLIDATING TOKENIZATION, ENCODING AND AGGREGATION INTO ONE FUNCTION
'''
context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

def encode_contexts(text_list):
    # Encode a list of texts into embeddings
    embeddings = []
    for text in text_list:
        inputs = context_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256)
        outputs = context_encoder(**inputs)
        embeddings.append(outputs.pooler_output)
    return torch.cat(embeddings).detach().numpy()

# encode imported doc paragraphs to create embeddings.
context_embeddings = encode_contexts(paragraphs)
