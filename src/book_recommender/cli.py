"""CLI entry point for quick smoke runs."""

from __future__ import annotations

import argparse

from .data_loader import load_datasets
from .recommender import BookRecommender


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Book recommender system using k-NN.")
    parser.add_argument("--recommend-user", type=int, help="User ID for top-N recommendations.")
    parser.add_argument("--predict-user", type=int, help="User ID for rating prediction.")
    parser.add_argument("--predict-isbn", type=str, help="ISBN for rating prediction.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of recommendations to return.")
    parser.add_argument(
        "--min-user-ratings",
        type=int,
        default=200,
        help="Minimum ratings required for user to stay in training data.",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=10,
        help="Maximum number of nearest neighbors to use.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    books, users, ratings = load_datasets()
    recommender = BookRecommender(
        books=books,
        users=users,
        ratings=ratings,
        min_user_ratings=args.min_user_ratings,
        n_neighbors=args.neighbors,
    )

    if args.recommend_user is not None:
        recommendations = recommender.recommend_books_for_user(args.recommend_user, top_n=args.top_n)
        print(f"Top {args.top_n} recommendations for user {args.recommend_user}:")
        for title in recommendations:
            print(f"- {title}")

    if args.predict_user is not None and args.predict_isbn is not None:
        prediction = recommender.predict_book_rating(args.predict_user, args.predict_isbn)
        print(f"Prediction for user {args.predict_user}, ISBN {args.predict_isbn}:")
        print(prediction)

    if args.recommend_user is None and (args.predict_user is None or args.predict_isbn is None):
        raise SystemExit("Provide --recommend-user or both --predict-user and --predict-isbn.")


if __name__ == "__main__":
    main()
