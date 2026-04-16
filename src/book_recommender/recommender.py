"""Core recommendation engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


@dataclass
class BookRecommender:
    books: pd.DataFrame
    users: pd.DataFrame
    ratings: pd.DataFrame
    min_user_ratings: int = 200
    n_neighbors: int = 10

    def __post_init__(self) -> None:
        self.books = self._prepare_books(self.books)
        self.filtered_ratings = self._filter_active_users(self.ratings, self.min_user_ratings)
        self.final_rating = self._build_final_rating(self.filtered_ratings, self.books)
        self.user_book_matrix = self._build_user_book_matrix(self.final_rating)
        self.book_user_matrix = self._build_book_user_matrix(self.final_rating)
        self.user_model = self._fit_model(self.user_book_matrix)
        self.book_model = self._fit_model(self.book_user_matrix)

    @staticmethod
    def _prepare_books(books: pd.DataFrame) -> pd.DataFrame:
        return books[["ISBN", "Book-Title", "Book-Author", "Year-Of-Publication", "Publisher"]].copy()

    @staticmethod
    def _filter_active_users(ratings: pd.DataFrame, min_user_ratings: int) -> pd.DataFrame:
        active_users = ratings["User-ID"].value_counts()
        active_user_ids = active_users[active_users > min_user_ratings].index
        return ratings[ratings["User-ID"].isin(active_user_ids)].copy()

    @staticmethod
    def _build_final_rating(ratings: pd.DataFrame, books: pd.DataFrame) -> pd.DataFrame:
        ratings_with_books = ratings.merge(books, on="ISBN")
        num_rating = (
            ratings_with_books.groupby("Book-Title")["Book-Rating"]
            .count()
            .reset_index()
            .rename(columns={"Book-Rating": "Number-of-Rating"})
        )
        final_rating = ratings_with_books.merge(num_rating, on="Book-Title")
        final_rating = final_rating.drop_duplicates(["User-ID", "Book-Title"])
        return final_rating

    @staticmethod
    def _build_user_book_matrix(final_rating: pd.DataFrame) -> pd.DataFrame:
        return final_rating.pivot_table(
            columns="Book-Title",
            index="User-ID",
            values="Book-Rating",
            fill_value=0,
        )

    @staticmethod
    def _build_book_user_matrix(final_rating: pd.DataFrame) -> pd.DataFrame:
        return final_rating.pivot_table(
            columns="User-ID",
            index="ISBN",
            values="Book-Rating",
            fill_value=0,
        )

    def _fit_model(self, matrix: pd.DataFrame) -> NearestNeighbors:
        if matrix.empty:
            raise ValueError("Training matrix empty. Check dataset or filtering threshold.")

        neighbor_count = min(self.n_neighbors, len(matrix))
        model = NearestNeighbors(n_neighbors=neighbor_count, metric="cosine")
        model.fit(matrix.values)
        return model

    @staticmethod
    def _format_rating(value: float) -> str:
        return str(int(value)) if float(value).is_integer() else f"{value:.2f}"

    def recommend_books_for_user(self, user_id: int, top_n: int = 10) -> list[str]:
        if user_id not in self.user_book_matrix.index:
            raise ValueError(f"User ID {user_id} not found in recommendation matrix.")

        user_index = self.user_book_matrix.index.get_loc(user_id)
        _, suggestions = self.user_model.kneighbors(
            self.user_book_matrix.iloc[user_index].values.reshape(1, -1)
        )

        recommended_books: set[str] = set()
        for similar_user_index in suggestions.flatten():
            rated_books = self.user_book_matrix.iloc[similar_user_index]
            recommended_books.update(rated_books.index[rated_books > 0])

        target_user_rated = self.user_book_matrix.columns[self.user_book_matrix.iloc[user_index] > 0]
        recommended_books -= set(target_user_rated)

        ranked_books = sorted(
            recommended_books,
            key=lambda title: self.user_book_matrix.iloc[suggestions.flatten()][title].mean(),
            reverse=True,
        )
        return ranked_books[:top_n]

    def predict_book_rating(self, user_id: int, isbn: str) -> str:
        if user_id not in self.book_user_matrix.columns:
            raise ValueError(f"User ID {user_id} not found in rating matrix.")

        if isbn not in self.book_user_matrix.index:
            raise ValueError(f"ISBN {isbn} not found in rating matrix.")

        existing_rating = self.book_user_matrix.loc[isbn, user_id]
        if existing_rating > 0:
            return f"Already Rated. Rating: {self._format_rating(existing_rating)}"

        isbn_index = self.book_user_matrix.index.get_loc(isbn)
        distances, suggestions = self.book_model.kneighbors(
            self.book_user_matrix.iloc[isbn_index].values.reshape(1, -1)
        )

        weighted_ratings = []
        for distance, similar_book_index in zip(distances.flatten(), suggestions.flatten()):
            if similar_book_index == isbn_index:
                continue

            similar_book_isbn = self.book_user_matrix.index[similar_book_index]
            user_rating = self.book_user_matrix.loc[similar_book_isbn, user_id]
            if user_rating <= 0:
                continue

            similarity = 1 - distance
            if similarity <= 0:
                continue
            weighted_ratings.append((user_rating, similarity))

        if not weighted_ratings:
            return "No Predictions"

        weighted_sum = sum(rating * weight for rating, weight in weighted_ratings)
        weight_total = sum(weight for _, weight in weighted_ratings)
        predicted_rating = weighted_sum / weight_total
        return f"Rating: {self._format_rating(predicted_rating)}"
