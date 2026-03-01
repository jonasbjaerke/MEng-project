

from pathlib import Path
from .text_processing import TextFeaturePipeline
from ..utils import get_json


def run():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    posts_path = PROJECT_ROOT / "data" / "raw" / "posts" / "posts_mini.json"
    users_path = PROJECT_ROOT / "data" / "raw" / "users" / "userdata_mini.json"

    posts = get_json(posts_path)
    users = get_json(users_path)

    pipeline = TextFeaturePipeline()

    pipeline.run(posts, users)

    print("Text feature pipeline complete.")


if __name__ == "__main__":
    run()