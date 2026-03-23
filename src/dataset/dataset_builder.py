

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

        self._hashtag_users = None
        self._hashtag_posts = None

    @abstractmethod
    def build_features(self, A_id, S_id, P_id, post, label):
        pass

    def _build_hashtag_caches(self):
        hashtag_users = {}
        hashtag_posts = {}

        for P_id, post in self.posts.items():
            hashtag = post.get("hashtag")
            if not hashtag:
                continue

            sender = (post.get("author") or {}).get("did")
            reposters = post.get("stored_reposters") or []

            if hashtag not in hashtag_users:
                hashtag_users[hashtag] = set()
            if hashtag not in hashtag_posts:
                hashtag_posts[hashtag] = []

            if reposters == []: #temp
                hashtag_posts[hashtag].append((P_id, post))

            if sender:
                hashtag_users[hashtag].add(sender)

            hashtag_users[hashtag].update(reposters)

        self._hashtag_users = hashtag_users
        self._hashtag_posts = hashtag_posts




    def build(self, neg_per_pos: int = 1) -> pd.DataFrame:

        if self._hashtag_users is None or self._hashtag_posts is None:
            self._build_hashtag_caches()

        rows = []

        for P_id, post in tqdm(self.posts.items(), desc="Building dataset"):

            S_id = (post.get("author") or {}).get("did")
            if not S_id or S_id not in self.users:
                continue

            stored_reposters = post.get("stored_reposters") or []
            if not stored_reposters:
                continue

            hashtag = post.get("hashtag")
            if not hashtag:
                continue

            candidate_pool = self._hashtag_users.get(hashtag)
            hashtag_posts = self._hashtag_posts.get(hashtag)

            if not candidate_pool or not hashtag_posts:
                continue

            # -------- Positive --------
            A_id = stored_reposters[0]

            if A_id not in self.users:
                continue

            pos_row = self.build_features(A_id, S_id, P_id, post, 1)
            if not pos_row:
                continue
            rows.append(pos_row)

            # -------- Negative --------
            neg_indices = self.rng.choice(
                len(hashtag_posts),
                size=min(neg_per_pos, len(hashtag_posts)),
                replace=False
            )

            for idx in np.atleast_1d(neg_indices):
                neg_P_id, neg_post = hashtag_posts[int(idx)]

                neg_S_id = (neg_post.get("author") or {}).get("did")
                if not neg_S_id or neg_S_id not in self.users:
                    continue

                neg_reposters = neg_post.get("reposted_by") or []

                neg_pool = list(
                    candidate_pool
                    - {neg_S_id}
                    - set(neg_reposters)
                )

                if not neg_pool:
                    continue

                neg_A = self.rng.choice(neg_pool)

                if neg_A not in self.users:
                    continue

                neg_row = self.build_features(neg_A, neg_S_id, neg_P_id, neg_post, 0)
                if neg_row:
                    rows.append(neg_row)

        return pd.DataFrame(rows)

    # def build(self, neg_per_pos: int = 1) -> pd.DataFrame:
    #     if self._hashtag_users is None or self._hashtag_posts is None:
    #         self._build_hashtag_caches()

    #     rows = []
    #     used_post_ids = set()

    #     for P_id, post in tqdm(self.posts.items(), desc="Building dataset"):

    #         # Skip if this post was already used earlier
    #         # either as a positive or as a negative
    #         if P_id in used_post_ids:
    #             continue

    #         hashtag = post.get("hashtag")
    #         if not hashtag:
    #             continue

    #         candidate_pool = self._hashtag_users.get(hashtag)
    #         hashtag_posts = self._hashtag_posts.get(hashtag)

    #         if not candidate_pool or not hashtag_posts:
    #             continue

    #         S_id = (post.get("author") or {}).get("did")
    #         if not S_id or S_id not in self.users:
    #             continue

    #         stored_reposters = post.get("stored_reposters") or []
    #         if not stored_reposters:
    #             continue

    #         # -------- Positive --------
    #         A_id = stored_reposters[0]

    #         if A_id not in self.users:
    #             continue

    #         pos_row = self.build_features(A_id, S_id, P_id, post, 1)
    #         if pos_row:
    #             rows.append(pos_row)
    #             used_post_ids.add(P_id)

    #         # -------- Negative --------
    #         available_neg_posts = [
    #             (neg_P_id, neg_post)
    #             for neg_P_id, neg_post in hashtag_posts
    #             if neg_P_id not in used_post_ids
    #             and not (neg_post.get("stored_reposters") or [])
    #         ]

    #         if not available_neg_posts:
    #             continue

    #         n_to_sample = min(neg_per_pos, len(available_neg_posts))
    #         neg_indices = self.rng.choice(
    #             len(available_neg_posts),
    #             size=n_to_sample,
    #             replace=False
    #         )

    #         for idx in np.atleast_1d(neg_indices):
    #             neg_P_id, neg_post = available_neg_posts[int(idx)]

    #             neg_S_id = (neg_post.get("author") or {}).get("did")
    #             if not neg_S_id or neg_S_id not in self.users:
    #                 continue

    #             neg_reposters = neg_post.get("reposted_by") or []

    #             neg_pool = list(
    #                 candidate_pool
    #                 - {neg_S_id}
    #                 - set(neg_reposters)
    #             )

    #             if not neg_pool:
    #                 continue

    #             neg_A = self.rng.choice(neg_pool)

    #             if neg_A not in self.users:
    #                 continue

    #             neg_row = self.build_features(neg_A, neg_S_id, neg_P_id, neg_post, 0)
    #             if neg_row:
    #                 rows.append(neg_row)
    #                 used_post_ids.add(neg_P_id)

    #     return pd.DataFrame(rows)

    @abstractmethod
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    