import re
import torch
from transformers import AutoTokenizer, AutoModel


class MessageBertFeatureExtractor:
    """
    Extracts message-level BERT features for the current post P only.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 128,
        pooling: str = "cls",
        device: str | None = None,
        normalize_urls: bool = True,
        normalize_mentions: bool = False,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.pooling = pooling
        self.normalize_urls = normalize_urls
        self.normalize_mentions = normalize_mentions
        self.cache: dict[str, dict[str, float]] = {}

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def get_post_text(self, post: dict) -> str:
        """
        Extract text from Bluesky-style post structure.
        """
        text = post.get("record", {}).get("text")
        if text is None:
            return ""
        return str(text)

    def prepare_text(self, text: str) -> str:
        """
        Light preprocessing only.
        Keep text close to original so BERT can use it naturally.
        """
        text = text.strip()

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        if self.normalize_urls:
            text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)

        if self.normalize_mentions:
            text = re.sub(r"@\w+", "[USER]", text)

        return text

    @torch.no_grad()
    def encode_text(self, text: str) -> list[float]:
        """
        Encode one text string into a dense BERT vector.
        """
        if not text:
            hidden_size = self.model.config.hidden_size
            return [0.0] * hidden_size

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        last_hidden = outputs.last_hidden_state  # [1, seq_len, hidden_size]

        if self.pooling == "cls":
            vec = last_hidden[:, 0, :]
        elif self.pooling == "mean":
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            masked_hidden = last_hidden * attention_mask
            summed = masked_hidden.sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1)
            vec = summed / counts
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling}")

        return vec.squeeze(0).cpu().tolist()

    def get_post_features(self, post_id: str, post: dict) -> dict[str, float]:
        """
        Return cached flat BERT features for one post.
        """
        if post_id in self.cache:
            return self.cache[post_id]

        raw_text = self.get_post_text(post)
        clean_text = self.prepare_text(raw_text)
        embedding = self.encode_text(clean_text)

        features = {f"BERT_{i}": val for i, val in enumerate(embedding)}
        self.cache[post_id] = features
        return features

    def calc_features(self, A_id, S_id, P_id, post, label):
        row = {
            "A_id": A_id,
            "S_id": S_id,
            "P_id": P_id,
            "hashtag": post.get("hashtag"),
            "label": label,
        }

        p_features = self.get_post_features(P_id, post)
        for k, v in p_features.items():
            row[f"M-P_{k}"] = v

        return row