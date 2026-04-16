from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from book_recommender.recommender import BookRecommender


def build_fixture() -> BookRecommender:
    books = pd.DataFrame(
        [
            {"ISBN": "A", "Book-Title": "Book A", "Book-Author": "Author 1", "Year-Of-Publication": 2000, "Publisher": "Pub"},
            {"ISBN": "B", "Book-Title": "Book B", "Book-Author": "Author 2", "Year-Of-Publication": 2001, "Publisher": "Pub"},
            {"ISBN": "C", "Book-Title": "Book C", "Book-Author": "Author 3", "Year-Of-Publication": 2002, "Publisher": "Pub"},
        ]
    )
    users = pd.DataFrame(
        [
            {"User-ID": 1, "Location": "A", "Age": 20},
            {"User-ID": 2, "Location": "B", "Age": 21},
            {"User-ID": 3, "Location": "C", "Age": 22},
        ]
    )
    ratings = pd.DataFrame(
        [
            {"User-ID": 1, "ISBN": "A", "Book-Rating": 10},
            {"User-ID": 1, "ISBN": "B", "Book-Rating": 8},
            {"User-ID": 2, "ISBN": "A", "Book-Rating": 9},
            {"User-ID": 2, "ISBN": "B", "Book-Rating": 7},
            {"User-ID": 2, "ISBN": "C", "Book-Rating": 6},
            {"User-ID": 3, "ISBN": "A", "Book-Rating": 8},
            {"User-ID": 3, "ISBN": "C", "Book-Rating": 9},
        ]
    )
    return BookRecommender(books=books, users=users, ratings=ratings, min_user_ratings=0, n_neighbors=2)


def test_recommend_books_for_user_returns_unrated_book() -> None:
    recommender = build_fixture()
    recommendations = recommender.recommend_books_for_user(1, top_n=3)
    assert "Book C" in recommendations


def test_predict_book_rating_uses_similar_books() -> None:
    recommender = build_fixture()
    prediction = recommender.predict_book_rating(1, "C")
    assert prediction.startswith("Rating:")


def test_predict_book_rating_returns_existing_rating() -> None:
    recommender = build_fixture()
    prediction = recommender.predict_book_rating(1, "A")
    assert prediction == "Already Rated. Rating: 10"


def test_recommender_rejects_zero_neighbors() -> None:
    with pytest.raises(ValueError, match="n_neighbors must be at least 1"):
        BookRecommender(
            books=build_fixture().books,
            users=build_fixture().users,
            ratings=build_fixture().ratings,
            min_user_ratings=0,
            n_neighbors=0,
        )


def test_recommend_books_for_user_rejects_non_positive_top_n() -> None:
    recommender = build_fixture()
    with pytest.raises(ValueError, match="top_n must be at least 1"):
        recommender.recommend_books_for_user(1, top_n=0)
