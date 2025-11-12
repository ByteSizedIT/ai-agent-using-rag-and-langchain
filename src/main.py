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
