
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

    def calc_features(self, A_id, S_id, P_id, post, label):
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
            row[f"M-{k}"] = v

        return row