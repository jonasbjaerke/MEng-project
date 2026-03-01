from datetime import datetime
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class HashtagDownloader:

    def __init__(self, api):
        self.api = api

    def fetch_hashtag(
        self,
        hashtag,
        since_dt,
        until_dt,
        max_posts=None,
        out_dir= PROJECT_ROOT / "data" / "raw" / "hashtags"
    ):
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
                    "since": since_dt.isoformat(),
                    "until": current_until.isoformat(),
                    "lang": "en"
                }

                r = self.api.get(
                    "app.bsky.feed.searchPosts",
                    params
                )

                if r.status_code != 200:
                    break

                posts = r.json().get("posts", [])
                if not posts:
                    break

                for p in posts:
                    f.write(json.dumps(p) + "\n")
                    total += 1
                    if max_posts and total >= max_posts:
                        break

                oldest = posts[-1]["record"]["createdAt"]
                current_until = datetime.fromisoformat(
                    oldest.replace("Z", "+00:00")
                )

                if max_posts and total >= max_posts:
                    break

        return total