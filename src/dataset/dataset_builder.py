from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from tqdm import tqdm


# class DatasetBuilder(ABC):
#     """
#     Abstract dataset builder.
#     Handles sampling logic.
#     Subclasses define feature extraction.
#     """

#     def __init__(self, users: dict, posts: dict):
#         self.users = users
#         self.posts = posts
#         self.all_users = set(users.keys())
#         self.rng = np.random.default_rng()

#         self._hashtag_users = None
#     # ---------------------------------------
#     # Subclasses must implement this
#     # ---------------------------------------
#     @abstractmethod
#     def build_features(self, A_id, S_id, P_id, post, label):
#         pass

    
#     def _build_hashtag_user_cache(self):

#         hashtag_users = {}

#         for post in self.posts.values():

#             hashtag = post.get("hashtag")
#             if not hashtag:
#                 continue

#             sender = (post.get("author") or {}).get("did")
#             reposters = post.get("stored_reposters") or []

#             if hashtag not in hashtag_users:
#                 hashtag_users[hashtag] = set()

#             if sender:
#                 hashtag_users[hashtag].add(sender)

#             hashtag_users[hashtag].update(reposters)

#         self._hashtag_users = hashtag_users


#     def build(self, neg_per_pos: int = 1) -> pd.DataFrame:

#         if self._hashtag_users is None:
#             self._build_hashtag_user_cache()

#         rows = []

#         for P_id, post in tqdm(self.posts.items(), desc="Building dataset"):

#             hashtag = post.get("hashtag")
#             if not hashtag:
#                 continue

#             candidate_pool = self._hashtag_users.get(hashtag)
#             if not candidate_pool:
#                 continue

#             S_id = (post.get("author") or {}).get("did")
#             if not S_id or S_id not in self.users:
#                 continue

#             reposted_by = post.get("reposted_by") or []
#             stored_reposters = post.get("stored_reposters") or []

#             if not stored_reposters:
#                 continue

#             # -------- Positive --------
#             A_id = stored_reposters[0]

#             if A_id not in self.users:
#                 continue

#             pos_row = self.build_features(A_id, S_id, P_id, post, 1)
#             if pos_row:
#                 rows.append(pos_row)

#             # -------- Negative --------
#             neg_pool = list(
#                 candidate_pool
#                 - {S_id}
#                 - set(reposted_by)
#             )

#             if not neg_pool:
#                 continue

#             negs = self.rng.choice(
#                 neg_pool,
#                 size=min(neg_per_pos, len(neg_pool)),
#                 replace=False
#             )

#             for neg_A in negs:

#                 if neg_A not in self.users:
#                     continue

#                 neg_row = self.build_features(neg_A, S_id, P_id, post, 0)
#                 if neg_row:
#                     rows.append(neg_row)

#         return pd.DataFrame(rows)
    

#     @staticmethod
#     def remove_pair_duplicates(df: pd.DataFrame) -> pd.DataFrame:
#         return df.drop_duplicates(subset=["S_id", "A_id"])



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

            hashtag = post.get("hashtag")
            if not hashtag:
                continue

            candidate_pool = self._hashtag_users.get(hashtag)
            hashtag_posts = self._hashtag_posts.get(hashtag)

            if not candidate_pool or not hashtag_posts:
                continue

            S_id = (post.get("author") or {}).get("did")
            if not S_id or S_id not in self.users:
                continue

            stored_reposters = post.get("stored_reposters") or []
            if not stored_reposters:
                continue

            # -------- Positive --------
            A_id = stored_reposters[0]

            if A_id not in self.users:
                continue

            pos_row = self.build_features(A_id, S_id, P_id, post, 1)
            if pos_row:
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

    @staticmethod
    def remove_pair_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates(subset=["S_id", "A_id"])