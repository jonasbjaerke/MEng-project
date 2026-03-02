# src/data/runner.py

from datetime import datetime
from .pipeline import DataPipeline


if __name__ == "__main__":

    hashtags = [
        "Pokemon",
        "TheTraitors",
        "OpenAI",
        "Booksky",
        "Anime",
        "Iran",
        "ICE"
    ]

    since_dt = datetime(2026, 1, 15)
    until_dt = datetime(2026, 3, 2)

    max_posts_per_hashtag = 100
    min_posts_per_hashtag = 50

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