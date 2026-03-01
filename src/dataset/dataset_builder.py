# src/features/dataset_builder.py

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from tqdm import tqdm


class DatasetBuilder(ABC):
    """
    Abstract dataset builder.
    Handles sampling logic.
    Subclasses define feature extraction.
    """

    def __init__(self, users: dict, posts: dict):
        self.users = users
        self.posts = posts
        self.all_users = set(users.keys())
        self.rng = np.random.default_rng()

    # ---------------------------------------
    # Subclasses must implement this
    # ---------------------------------------
    @abstractmethod
    def build_features(self, A_id, S_id, P_id, post, label):
        pass

    # ---------------------------------------
    # Shared dataset logic
    # ---------------------------------------
    def build(self, neg_per_pos: int = 1) -> pd.DataFrame:

        rows = []

        for P_id, post in tqdm(self.posts.items(), desc="Building dataset"):

            S_id = post.get("author", {}).get("did")
            if not S_id or S_id not in self.users:
                continue

            reposted_by = post.get("reposted_by") or []
            if not reposted_by:
                continue

            # -------- Positive --------
            A_id = reposted_by[0]

            if A_id not in self.users:
                continue

            pos_row = self.build_features(A_id, S_id, P_id, post, 1)
            if pos_row:
                rows.append(pos_row)

            # -------- Negative --------
            neg_pool = list(self.all_users - {S_id} - set(reposted_by))
            if not neg_pool:
                continue

            negs = self.rng.choice(
                neg_pool,
                size=min(neg_per_pos, len(neg_pool)),
                replace=False
            )

            for neg_A in negs:
                neg_row = self.build_features(neg_A, S_id, P_id, post, 0)
                if neg_row:
                    rows.append(neg_row)

        return pd.DataFrame(rows)

    @staticmethod
    def remove_pair_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates(subset=["S_id", "A_id"])