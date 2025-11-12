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