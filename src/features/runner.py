from pathlib import Path
from typing import Type

from src.utils import get_json, save_csv
from src.features.dataset_builder import DatasetBuilder


RAW_BASE = Path("data/raw")
POSTS_DIR = RAW_BASE / "posts"
USERS_DIR = RAW_BASE / "users"
PROCESSED_DIR = Path("data/processed")


def build_dataset(
    builder_cls: Type[DatasetBuilder],   # 👈 pass concrete builder
    posts_filename: str = "posts",
    users_filename: str = "users",
    output_filename: str = "dataset.csv",
    neg_per_pos: int = 1,
):
    """
    Build ML dataset using a specific DatasetBuilder.
    """

    posts_path = POSTS_DIR / posts_filename
    users_path = USERS_DIR / users_filename
    output_path = PROCESSED_DIR / output_filename

    # -------------------------
    # Load raw data
    # -------------------------
    posts = get_json(posts_path)
    users = get_json(users_path)

    # -------------------------
    # Build dataset
    # -------------------------
    builder = builder_cls(users, posts)
    df = builder.build(neg_per_pos=neg_per_pos)

    df = builder.remove_pair_duplicates(df)

    # -------------------------
    # Save processed dataset
    # -------------------------
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    save_csv(df, output_path)

    print(f"Dataset saved → {output_path}")
    print(f"Rows: {len(df)}")

    return df