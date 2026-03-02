# src/data/pipeline.py

from pathlib import Path
from datetime import datetime
import asyncio

from .api import BlueskyAPI
from .downloader import HashtagDownloader
from .users import UserDataCollector
from ..utils import write_json
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class DataPipeline:
    """
    High-level orchestration of the full data collection pipeline.
    """

    def __init__(
        self,
        hashtags,
        since_dt,
        until_dt,
        max_posts_per_hashtag,
        min_posts_per_hashtag,
        base_data_dir= PROJECT_ROOT / "data"
    ):
        self.hashtags = hashtags
        self.since_dt = since_dt
        self.until_dt = until_dt
        self.max_posts_per_hashtag = max_posts_per_hashtag
        self.min_posts_per_hashtag = min_posts_per_hashtag

        self.base_dir = base_data_dir
        self.raw_dir = self.base_dir / "raw"
        self.hashtag_dir = self.raw_dir / "hashtags"
        self.users_dir = self.raw_dir / "users"
        self.posts_dir = self.raw_dir / "posts"
        self.texts_dir = self.users_dir / "texts"

        self.hashtag_dir.mkdir(parents=True, exist_ok=True)
        self.users_dir.mkdir(parents=True, exist_ok=True)

        self.api = BlueskyAPI()
        self.downloader = HashtagDownloader(self.api)

    # -----------------------------
    # Stage 1: Download posts
    # -----------------------------
    def download_posts(self):
        print("Downloading hashtag posts...")

        for tag in self.hashtags:
            self.downloader.fetch_hashtag(
                hashtag=tag,
                since_dt=self.since_dt,
                until_dt=self.until_dt,
                max_posts=self.max_posts_per_hashtag,
                out_dir=self.hashtag_dir
            )

        print("Finished downloading.")

    # -----------------------------
    # Stage 2: Load posts
    # -----------------------------
    def load_posts(self):
        print("Loading posts into memory...")

        files = list(self.hashtag_dir.glob("*.jsonl"))
        posts = {}

        for path in files:
            path = Path(path)
            hashtag = path.stem  # filename without extension

            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        post = json.loads(line.strip())
                    except Exception:
                        continue

                    uri = post.get("uri")
                    if not uri:
                        continue

                    post["hashtag"] = hashtag
                    posts[uri] = post

        print(f"Loaded {len(posts)} posts.")
        return posts

    # -----------------------------
    # Stage 3: Build users
    # -----------------------------
    async def build_users(self, posts):
        print("Collecting user data...")

        collector = UserDataCollector(posts)
        users = await collector.collect()

        return users
    
    
    # -----------------------------
    # Run entire pipeline
    # -----------------------------
    def run(self):

        # Step 1: download
        self.download_posts()

        # Step 2: load posts
        posts = self.load_posts()

        # Step 3: build users (async)
        users = asyncio.run(self.build_users(posts))


        # Step 4: save users and posts
        users_path = self.users_dir / "users.json"
        write_json(users, users_path)

        posts_path = self.posts_dir / "posts.json"
        write_json(posts, posts_path)


        print("Collecting data complete.")