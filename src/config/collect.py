from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence


@dataclass
class CollectionConfig:
    """
    Settings that define the raw data collection process.
    """
    hashtags: Sequence[str] = field(default_factory=lambda: [
        "AI",
        "Anime",
        "BlackHistoryMonth",
        "Booksky",
        "Gaza",
        "ICE",
        "Pokemon",
        "Superbowl",
        "TheTraitors",
        "Trump",
    ])

    since_dt: datetime = datetime(2026, 1, 15)
    until_dt: datetime = datetime(2026, 3, 2)

    max_posts_per_hashtag: int = 11000
    min_posts_per_hashtag: int = 9000

