import importlib.metadata
from .training import train_bpe
from .tokenizer import Tokenizer

try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    pass

__all__ = ["train_bpe", "Tokenizer"]