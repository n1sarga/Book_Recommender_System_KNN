# Book Recommender System KNN
This project implements a k-Nearest Neighbors based book recommendation system using the Book-Crossing dataset.

## Project Structure

```text
Book_Recommender_System_KNN/
|-- data/
|   `-- raw/
|       |-- BX_Books.csv
|       |-- BX-Users.csv
|       `-- BX-Book-Ratings.csv
|-- scripts/
|   `-- run_demo.py
|-- src/
|   `-- book_recommender/
|       |-- __init__.py
|       |-- cli.py
|       |-- config.py
|       |-- data_loader.py
|       `-- recommender.py
|-- tests/
|   `-- test_recommender.py
|-- .gitignore
|-- pyproject.toml
|-- requirements.txt
`-- README.md
```

## Features

- User-based k-NN recommendations for unread books.
- Book-based rating prediction using similar books user already rated.
- Configurable neighbor count and active-user threshold.
- Lightweight test coverage for core behaviors.

## How It Works

- `scripts/run_demo.py` starts program from terminal.
- `src/book_recommender/cli.py` reads command-line arguments and decides whether to recommend books or predict a rating.
- `src/book_recommender/config.py` defines where dataset files are stored.
- `src/book_recommender/data_loader.py` loads book, user, and rating CSV files into pandas DataFrames.
- `src/book_recommender/recommender.py` filters active users, prepares rating data, builds two pivot tables, and trains k-NN models with cosine similarity.
- One matrix is used to find similar users for book recommendations, and the other is used to find similar books for rating prediction.
- `tests/test_recommender.py` checks that core recommendation and prediction behavior works as expected.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Recommend books for user:

```bash
python scripts/run_demo.py --recommend-user 254 --top-n 10
```

Predict rating for user and ISBN:

```bash
python scripts/run_demo.py --predict-user 277427 --predict-isbn 002542730X
```

Run tests:

```bash
pytest
```

## Data

Raw CSV files stored in `data/raw/`. Loader expects Book-Crossing dataset filenames already in repo.
