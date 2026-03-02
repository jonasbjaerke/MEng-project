from datetime import datetime, timezone
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def fmt(dt: datetime) -> str:
    """Convert datetime → Bluesky timestamp format."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def parse_dt(created: str) -> datetime:
    """Parse Bluesky createdAt → naive UTC datetime."""
    return (
        datetime.fromisoformat(created.replace("Z", "+00:00"))
        .astimezone(timezone.utc)
        .replace(tzinfo=None)
    )


class HashtagDownloader:
    def __init__(self, api):
        self.api = api

    def fetch_hashtag(
        self,
        hashtag: str,
        since_dt: datetime,
        until_dt: datetime,
        max_posts: int | None = None,
        min_posts: int | None = None,
        out_dir: Path = PROJECT_ROOT / "data" / "raw" / "hashtags",
    ) -> int:
        """
        Fetch all posts matching a hashtag between since_dt and until_dt.
        Steps backward in time using the oldest timestamp returned.
        Raises ValueError if fewer than min_posts are collected.
        """

        out_dir.mkdir(parents=True, exist_ok=True)
        outfile = out_dir / f"{hashtag}.jsonl"

        total = 0
        current_until = until_dt

        with outfile.open("w", encoding="utf-8") as f:

            while True:
                params = {
                    "q": f"#{hashtag}", 
                    "limit": 100,
                    "sort": "latest",
                    "since": fmt(since_dt),
                    "until": fmt(current_until),
                    "lang": "en",
                }

                r = self.api.get("app.bsky.feed.searchPosts", params)

                if r.status_code != 200:
                    print("Search failed:", r.status_code, r.text)
                    break

                posts = r.json().get("posts", [])
                if not posts:
                    break

                for p in posts:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
                    total += 1

                    if max_posts and total >= max_posts:
                        break

                # Oldest post in this batch
                oldest_ts = posts[-1]["record"]["createdAt"]
                oldest_dt = parse_dt(oldest_ts)

                # Stop if we reached lower time bound
                if oldest_dt <= since_dt:
                    break

                # Stop if we reached max_posts
                if max_posts and total >= max_posts:
                    break

                current_until = oldest_dt

        # Enforce minimum
        if min_posts is not None and total < min_posts:
            outfile.unlink(missing_ok=True)
            raise ValueError(
                f"Only fetched {total} posts for '{hashtag}', "
                f"which is less than required minimum ({min_posts})."
            )

        return total