from typing import Type
import argparse

from ..config import DATASET_CONFIGS, PathsConfig
from ..utils import get_json, save_csv
from .dataset_builder import DatasetBuilder
from .message_dataset_builder import MessageDatasetBuilder
from .user_dataset_builder import UserDatasetBuilder
from .hybrid_dataset_builder import HybridDatasetBuilder
from .bert_dataset_builder import MessageBertDatasetBuilder


FEATURE_REGISTRY = {
    "M": MessageDatasetBuilder,
    "U": UserDatasetBuilder,
    "Hybrid": HybridDatasetBuilder,
    "Bert": MessageBertDatasetBuilder,
}


def new_dataset(
    builder_cls: Type[DatasetBuilder],
    dataset_cfg,
    paths_cfg: PathsConfig,
    config_name: str,
):
    posts_path = paths_cfg.posts_dir / dataset_cfg.posts_filename
    users_path = paths_cfg.users_dir / dataset_cfg.users_filename
    output_path = paths_cfg.datasets_dir / dataset_cfg.output_filename

    posts = get_json(posts_path)
    users = get_json(users_path)

    builder = builder_cls(users, posts)
    df = builder.build(neg_per_pos=dataset_cfg.neg_per_pos)
    df = builder.remove_duplicates(df)

    paths_cfg.datasets_dir.mkdir(parents=True, exist_ok=True)
    save_csv(df, output_path)

    print(f"Dataset config: {config_name}")
    print(f"Dataset saved → {output_path}")
    print(f"Rows: {len(df)}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ML dataset")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        choices=DATASET_CONFIGS.keys(),
        help="Named dataset config to use",
    )
    args = parser.parse_args()

    dataset_cfg = DATASET_CONFIGS[args.config]
    paths_cfg = PathsConfig()

    builder_cls = FEATURE_REGISTRY[dataset_cfg.builder]

    new_dataset(
        builder_cls=builder_cls,
        dataset_cfg=dataset_cfg,
        paths_cfg=paths_cfg,
        config_name=args.config,
    )