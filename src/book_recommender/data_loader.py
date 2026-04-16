"""Dataset loading helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import BOOKS_FILE, RATINGS_FILE, USERS_FILE


def _read_csv(file_path: Path) -> pd.DataFrame:
    return pd.read_csv(file_path, sep=";", on_bad_lines="skip", encoding="latin-1")


def load_datasets(
    books_path: Path = BOOKS_FILE,
    users_path: Path = USERS_FILE,
    ratings_path: Path = RATINGS_FILE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw Book-Crossing datasets."""

    books = _read_csv(books_path)
    users = _read_csv(users_path)
    ratings = _read_csv(ratings_path)
    return books, users, ratings
