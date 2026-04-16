"""Book recommender package."""

from .data_loader import load_datasets
from .recommender import BookRecommender

__all__ = ["BookRecommender", "load_datasets"]
