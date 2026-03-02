# src/data/runner.py

from datetime import datetime
from .pipeline import DataPipeline


if __name__ == "__main__":

    # -----------------------------
    # Configuration
    # -----------------------------

    hashtags = [
        "politics",
        "sports",
        "technology",
    ]

    since_dt = datetime(2024, 1, 1)
    until_dt = datetime(2024, 12, 31)

    max_posts_per_hashtag = 500
    min_posts_per_hashtag = 100

    # -----------------------------
    # Run pipeline
    # -----------------------------

    pipeline = DataPipeline(
        hashtags=hashtags,
        since_dt=since_dt,
        until_dt=until_dt,
        max_posts_per_hashtag=max_posts_per_hashtag,
        min_posts_per_hashtag=min_posts_per_hashtag,
    )

    pipeline.run()