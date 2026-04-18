from ..config.paths import PathsConfig
from ..utils import get_json
from .text_processing import TextFeaturePipeline


def run():
    paths_cfg = PathsConfig()

    posts_path = paths_cfg.posts_dir / "postsFinal.json"
    users_path = paths_cfg.users_dir / "usersFinal.json"

    posts = get_json(posts_path)
    users = get_json(users_path)

    pipeline = TextFeaturePipeline()
    pipeline.run(posts, users)

    print("Text feature pipeline complete.")


if __name__ == "__main__":
    run()