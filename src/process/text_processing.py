import gc
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from .message_features import add_all_m_features


class TextFeaturePipeline:
    """
    Handles:
    1. Building raw text JSONL
    2. Computing text feature parquet chunks
    3. Merging chunks into processed JSON
    """

    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[2]

        # Directories
        self.data_dir = self.project_root / "data"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"

        # Files
        self.temp_file = self.raw_dir / "temp_text.jsonl"
        self.raw_features_dir = self.raw_dir / "text_features"
        self.final_json = self.processed_dir / "text_features.json"

    # --------------------------------------------------
    # Step 1: Build raw text JSONL
    # --------------------------------------------------

    def build_text_jsonl(self, posts: dict, users: dict):
        """
        Stream {post_uri, text} to JSONL file.

        Includes:
        - main posts from `posts`
        - historical posts from each user in `users[did]["history"]`

        Safe for millions of rows.
        """

        self.temp_file.parent.mkdir(parents=True, exist_ok=True)

        seen = set()

        def write_row(f, uri, text):
            if not uri or uri in seen:
                return

            text = "" if text is None else str(text)
            json.dump({"post_uri": uri, "text": text}, f, ensure_ascii=False)
            f.write("\n")
            seen.add(uri)

        with self.temp_file.open("w", encoding="utf-8") as f:
            # Main collected posts
            for uri, post in posts.items():
                text = post.get("record", {}).get("text")
                write_row(f, uri, text)

            # History posts from each collected user
            for user in users.values():
                history = user.get("history") or []

                for item in history:
                    uri = item.get("post_uri")
                    text = item.get("text")
                    write_row(f, uri, text)

    # --------------------------------------------------
    # Step 2: Compute feature parquet chunks
    # --------------------------------------------------

    def run_feature_extraction(
        self,
        chunk_size: int = 10000,
        batch_size: int = 128,
    ):
        if not self.temp_file.exists():
            raise FileNotFoundError(f"Missing input file: {self.temp_file}")

        self.raw_features_dir.mkdir(parents=True, exist_ok=True)

        total_lines = sum(1 for _ in self.temp_file.open("r", encoding="utf-8"))
        total_chunks = (total_lines + chunk_size - 1) // chunk_size

        print(f"Total posts: {total_lines}, Total chunks: {total_chunks}")

        reader = pd.read_json(
            self.temp_file,
            lines=True,
            chunksize=chunk_size,
        )

        for i, chunk in enumerate(
            tqdm(reader, total=total_chunks, desc="Processing chunks")
        ):
            enriched = add_all_m_features(
                chunk,
                text_col="text",   # <-- compatibility fix
            )

            part_file = self.raw_features_dir / f"part_{i}.parquet"
            enriched.to_parquet(part_file, index=False)

            del enriched
            gc.collect()

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --------------------------------------------------
    # Step 3: Merge parquet chunks into final JSON
    # --------------------------------------------------

    def build_final_feature_dict(self):
        """
        Merge parquet chunks into single JSON dict:
        {post_uri: feature_dict}
        """

        parquet_files = sorted(self.raw_features_dir.glob("part_*.parquet"))

        if not parquet_files:
            print("No parquet files found.")
            return

        self.processed_dir.mkdir(parents=True, exist_ok=True)

        with self.final_json.open("w", encoding="utf-8") as f:
            f.write("{\n")
            first = True

            for file in parquet_files:
                df_chunk = pd.read_parquet(file)

                for _, row in df_chunk.iterrows():
                    post_uri = row["post_uri"]
                    row_dict = row.drop(
                        labels=["text", "post_uri"],
                        errors="ignore"
                    ).to_dict()

                    if not first:
                        f.write(",\n")
                    first = False

                    json.dump(post_uri, f, ensure_ascii=False)
                    f.write(": ")
                    json.dump(row_dict, f, ensure_ascii=False, allow_nan=True)

                del df_chunk
                gc.collect()

            f.write("\n}")

        print(f"Feature dict written to {self.final_json}")

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------

    def cleanup(self):
        """
        Remove temporary text JSONL file.
        """
        if self.temp_file.exists():
            self.temp_file.unlink()

    # --------------------------------------------------
    # Full pipeline
    # --------------------------------------------------

    def run(self, posts: dict, users: dict):
        self.build_text_jsonl(posts, users)
        self.run_feature_extraction()
        self.build_final_feature_dict()
        self.cleanup()