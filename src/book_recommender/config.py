from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

BOOKS_FILE = DATA_DIR / "BX_Books.csv"
USERS_FILE = DATA_DIR / "BX-Users.csv"
RATINGS_FILE = DATA_DIR / "BX-Book-Ratings.csv"
