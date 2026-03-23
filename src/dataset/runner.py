from pathlib import Path
from typing import Type
import argparse
from ..utils import get_json, save_csv
from .dataset_builder import DatasetBuilder
from .message_dataset_builder import MessageDatasetBuilder
from .user_dataset_builder import UserDatasetBuilder
from .hybrid_dataset_builder import HybridDatasetBuilder


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_BASE = PROJECT_ROOT / "data" / "raw"
POSTS = RAW_BASE / "posts"
USERS = RAW_BASE / "users"
DATASETS = PROJECT_ROOT / "data" / "processed" / "datasets"


FEATURE_REGISTRY = {
    "M": MessageDatasetBuilder,
    "U": UserDatasetBuilder,
    "Hybrid": HybridDatasetBuilder,
}


def new_dataset(
    builder_cls: Type[DatasetBuilder],   
    neg_per_pos: int,
    posts_filename: str,
    users_filename: str,
    output_filename: str,
):
    """
    Build ML dataset using a specific DatasetBuilder.
    """

    posts_path = POSTS / posts_filename
    users_path = USERS / users_filename
    output_path = DATASETS / output_filename

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
    DATASETS.mkdir(parents=True, exist_ok=True)
    save_csv(df, output_path)

    print(f"Dataset saved → {output_path}")
    print(f"Rows: {len(df)}")

    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Build ML dataset")

    parser.add_argument(
        "--builder",
        type=str,
        required=True,
        choices=FEATURE_REGISTRY.keys(),
        help="Builder version"
    )

    parser.add_argument("--neg_per_pos", type=int, default=1)
    parser.add_argument("--posts", type=str, default="postsFinal.json")
    parser.add_argument("--users", type=str, default="usersFinal.json")
    parser.add_argument("--output", type=str, default="dataset.csv")

    args = parser.parse_args()

    builder_cls = FEATURE_REGISTRY[args.builder]

    new_dataset(
        builder_cls=builder_cls,
        neg_per_pos=args.neg_per_pos,
        posts_filename=args.posts,
        users_filename=args.users,
        output_filename=args.output,
    )