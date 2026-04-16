"""CLI entry point for quick smoke runs."""

from __future__ import annotations

import argparse

from .data_loader import load_datasets
from .recommender import BookRecommender


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be 0 or greater")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Book recommender system using k-NN.")
    parser.add_argument("--recommend-user", type=int, help="User ID for top-N recommendations.")
    parser.add_argument("--predict-user", type=int, help="User ID for rating prediction.")
    parser.add_argument("--predict-isbn", type=str, help="ISBN for rating prediction.")
    parser.add_argument("--top-n", type=positive_int, default=10, help="Number of recommendations to return.")
    parser.add_argument(
        "--min-user-ratings",
        type=non_negative_int,
        default=200,
        help="Minimum ratings required for user to stay in training data.",
    )
    parser.add_argument(
        "--neighbors",
        type=positive_int,
        default=10,
        help="Maximum number of nearest neighbors to use.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.recommend_user is None and (args.predict_user is None or args.predict_isbn is None):
        parser.print_help()
        print("\nExamples:")
        print("python scripts/run_demo.py --recommend-user 254 --top-n 10")
        print("python scripts/run_demo.py --predict-user 277427 --predict-isbn 002542730X")
        return

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


if __name__ == "__main__":
    main()
