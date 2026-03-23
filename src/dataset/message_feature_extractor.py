# import json
# import os
# from collections import Counter
# from statistics import mean
# from ..utils import get_json
# from pathlib import Path

# PROJECT_ROOT = Path(__file__).resolve().parents[2]
# TEXTDICT = PROJECT_ROOT / "data" / "processed" / "text_features.json"

# class MessageFeatureExtractor:
#     """
#     Extracts message-level features for:
#         - Current post P
#         - Avg of last 50 historical posts for sender (S)
#         - Avg of last 50 historical posts for receiver (A)

#     Numerical features → mean
#     Non-numerical features → most frequent value
#     """

#     def __init__(self, users):
#         self.users = users

#         if not os.path.exists(TEXTDICT):
#             raise FileNotFoundError(f"{TEXTDICT} not found")

#         self.text_dict = get_json(TEXTDICT)

#     # =====================================================
#     # Helpers
#     # =====================================================

#     def get_post_features(self, post_id):
#         return self.text_dict.get(post_id, {})

#     def aggregate_features(self, feature_list):
#         """
#         Aggregates:
#             - Numerical → mean
#             - Non-numerical → mode
#         """

#         if not feature_list:
#             return {}

#         aggregated = {}
#         feature_keys = set().union(*feature_list)

#         for key in feature_keys:
#             values = [f[key] for f in feature_list if key in f]

#             if not values:
#                 continue

#             # Numerical → mean
#             if all(isinstance(v, (int, float)) for v in values):
#                 aggregated[key] = mean(values)

#             # Non-numerical → most frequent
#             else:
#                 aggregated[key] = Counter(values).most_common(1)[0][0]

#         return aggregated

#     def get_last_n_posts_features(self, history, exclude_post_id, n=50):
#         """
#         Collect features from last n historical posts (excluding P).
#         """

#         post_ids = [
#             h.get("post_uri")
#             for h in history
#             if h.get("post_uri") != exclude_post_id 
#             and h.get("parent_post_uri") != exclude_post_id
#         ]

#         post_ids = post_ids[-n:]  # last n

#         feature_list = []

#         for pid in post_ids:
#             if pid in self.text_dict:
#                 feature_list.append(self.text_dict[pid])

#         return self.aggregate_features(feature_list)

#     # =====================================================
#     # Main builder
#     # =====================================================

#     def build_features(self, A_id, S_id, P_id, post, label):
#         """
#         Returns message-level features for:
#             - Current post P
#             - Historical avg (last 50) for sender S
#             - Historical avg (last 50) for receiver A
#         """

#         A = self.users[A_id]
#         S = self.users[S_id]

#         row = {
#             "P_id": P_id,
#             "A_id": A_id,
#             "S_id": S_id,
#             "hashtag": post.get("hashtag"),
#             "label": label,
#         }

#         # -------------------------------------------------
#         # Current Post Features (P)
#         # -------------------------------------------------
#         P_features = self.get_post_features(P_id)

#         # Explicitly ensure hashtag from post overrides any dict value
#         P_features = {
#             **P_features,
#         }

    

#         for k, v in P_features.items():
#             row[f"M-P_{k}"] = v 

#         # -------------------------------------------------
#         # Reciever Historical Features (S)
#         # -------------------------------------------------
#         A_hist_features = self.get_last_n_posts_features(
#             A["history"],
#             P_id,
#         )

#         for k, v in A_hist_features.items():
#             row[f"M-H_R_{k}"] = v 

#         # -------------------------------------------------
#         # Sender Historical Features (A)
#         # -------------------------------------------------
#         S_hist_features = self.get_last_n_posts_features(
#             S["history"],
#             P_id,
#         )

#         for k, v in S_hist_features.items():
#             row[f"M-H_S_{k}"] = v 

#         return row

import os
from pathlib import Path

from ..utils import get_json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEXTDICT = PROJECT_ROOT / "data" / "processed" / "text_features.json"


class MessageFeatureExtractor:
    """
    Extracts message-level features only for the current post P.
    """

    def __init__(self):

        if not os.path.exists(TEXTDICT):
            raise FileNotFoundError(f"{TEXTDICT} not found")

        self.text_dict = get_json(TEXTDICT)

    def get_post_features(self, post_id):
        return self.text_dict.get(post_id, {})

    def build_features(self, A_id, S_id, P_id, post, label):
        """
        Returns message-level features for the current post P.
        """

        row = {
            "A_id": A_id,
            "S_id": S_id,
            "P_id": P_id,
            "hashtag": post.get("hashtag"),
            "label": label,
        }

        # Current post features
        P_features = self.get_post_features(P_id)
        if not P_features:
            print(f"Didn't find text features for {P_id}")

        for k, v in P_features.items():
            row[f"M-P_{k}"] = v

        return row