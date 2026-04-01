import re


class MessageBertClassifierExtractor:
    """
    Prepares message text for end-to-end BERT classification.
    This does NOT create embedding features.
    """

    def __init__(
        self,
        normalize_urls: bool = True,
        normalize_mentions: bool = False,
    ):
        self.normalize_urls = normalize_urls
        self.normalize_mentions = normalize_mentions

    def get_post_text(self, post: dict) -> str:
        text = post.get("record", {}).get("text")
        if text is None:
            return ""
        return str(text)

    def prepare_text(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"\s+", " ", text)

        if self.normalize_urls:
            text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)

        if self.normalize_mentions:
            text = re.sub(r"@\w+", "[USER]", text)

        return text

    def calc_features(self, A_id, S_id, P_id, post, label):
        raw_text = self.get_post_text(post)
        clean_text = self.prepare_text(raw_text)

        return {
            "A_id": A_id,
            "S_id": S_id,
            "P_id": P_id,
            "hashtag": post.get("hashtag"),
            "text": clean_text,
            "label": int(label),
        }