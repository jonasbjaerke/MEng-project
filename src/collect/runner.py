from .pipeline import DataPipeline
from ..config import CollectionConfig


if __name__ == "__main__":
    cfg = CollectionConfig()

    pipeline = DataPipeline(
        hashtags=list(cfg.hashtags),
        since_dt=cfg.since_dt,
        until_dt=cfg.until_dt,
        max_posts_per_hashtag=cfg.max_posts_per_hashtag,
        min_posts_per_hashtag=cfg.min_posts_per_hashtag,
    )

    pipeline.run()