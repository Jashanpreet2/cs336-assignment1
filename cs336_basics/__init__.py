import importlib.metadata
from .training import train_bpe
from .tokenizer import Tokenizer
from .transformer import *

try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    pass

__all__ = ["train_bpe", "Tokenizer", 
           "Linear", "Embedding", "RMSnorm", "FeedForward", "softmax",
           "scaled_dot_product_attention", "MultiHeadSelfAttention"]