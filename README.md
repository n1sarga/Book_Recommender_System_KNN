# Book Recommender System KNN

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
