# IMPORTANT: Much of this code is inspired from code written by Ziming Xu.


import os
import gc
import re
import time
from typing import List

import numpy as np
import pandas as pd
import nltk
import readability

try:
    import preprocessor as tweet_preprocessor
    PREPROCESSOR_AVAILABLE = True
except Exception:
    tweet_preprocessor = None
    PREPROCESSOR_AVAILABLE = False

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers.utils import logging
logging.set_verbosity_error()

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from pysentimiento import create_analyzer
from pysentimiento.preprocessing import preprocess_tweet
from scipy.special import softmax, expit


def ensure_nltk_resources():
    resources = [
        ("sentiment/vader_lexicon.zip", "vader_lexicon"),
        ("tokenizers/punkt", "punkt"),
    ]

    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


ensure_nltk_resources()

torch.set_float32_matmul_precision("high")

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("Using device:", DEVICE)


def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


tokenizer_irony, model_irony = load_model("cardiffnlp/twitter-roberta-base-irony")
tokenizer_offensive, model_offensive = load_model("cardiffnlp/twitter-roberta-base-offensive")
tokenizer_emoji, model_emoji = load_model("cardiffnlp/twitter-roberta-base-emoji")
tokenizer_topic, model_topic = load_model("cardiffnlp/tweet-topic-21-multi")
tokenizer_topic_single, model_topic_single = load_model("cardiffnlp/tweet-topic-21-single")

emotion_analyzer = create_analyzer(task="emotion", lang="en", device=DEVICE)
hate_speech_analyzer = create_analyzer(task="hate_speech", lang="en", device=DEVICE)

class_mapping = list(model_topic.config.id2label.values())
class_mapping_single = list(model_topic_single.config.id2label.values())

sia = SentimentIntensityAnalyzer()


def _clean_with_tweet_preprocessor(text: str) -> str:
    if PREPROCESSOR_AVAILABLE:
        tweet_preprocessor.set_options(
            tweet_preprocessor.OPT.URL,
            tweet_preprocessor.OPT.EMOJI,
            tweet_preprocessor.OPT.MENTION,
            tweet_preprocessor.OPT.NUMBER,
        )
        return tweet_preprocessor.clean(text)
    return text


def berkem_preprocess(text: str) -> str:
    text = str(text)
    text = re.sub(r"RT @\w+: ", " ", text)
    text = re.sub(r"(@[A-Za-z0-9_]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text)
    text = _clean_with_tweet_preprocessor(text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"&amp", "", text)
    return text.lower().strip()


def grammar_p(text: str) -> str:
    text = str(text)
    text = re.sub(r"RT @\w+: ", " ", text)
    text = re.sub(r"(@[A-Za-z0-9_]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text)
    text = _clean_with_tweet_preprocessor(text)
    return text


def roberta_preprocess(text: str) -> str:
    new_text = []
    for t in str(text).split():
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)


def assign_sentiment(neg: float, pos: float) -> str:
    if neg > pos:
        return "negative"
    elif pos > neg:
        return "positive"
    return "neutral"


def read_tokenized(text: str) -> str:
    text = str(text)
    try:
        sentences = nltk.sent_tokenize(text)
        return "\n".join(sentences)
    except Exception:
        return text


def safe_readability(tokenized_text: str):
    tokenized_text = str(tokenized_text).strip()

    if not tokenized_text:
        return None

    if not re.search(r"[A-Za-z]", tokenized_text):
        return None

    try:
        return readability.getmeasures(tokenized_text, lang="en")
    except ValueError as e:
        if "no words there" in str(e):
            return None
        raise


def build_preprocessed_texts(texts: List[str]) -> pd.DataFrame:
    rows = []

    for text in texts:
        raw = str(text)
        rows.append(
            {
                "full_text": raw,
                "text_berkem": berkem_preprocess(raw),
                "text_grammar": grammar_p(raw),
                "text_roberta": roberta_preprocess(raw),
                "text_tweet": preprocess_tweet(raw),
            }
        )

    return pd.DataFrame(rows)


def m_mapping_category_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    mapping_sentiment_overall = {"neutral": 0, "negative": 1, "positive": 2}
    mapping_emo_overall = {
        "others": 0,
        "joy": 1,
        "sadness": 2,
        "disgust": 3,
        "surprise": 4,
        "anger": 5,
        "fear": 6,
    }
    mapping_topic_overall = {
        "arts_&_culture": 0,
        "business_&_entrepreneurs": 1,
        "celebrity_&_pop_culture": 2,
        "diaries_&_daily_life": 3,
        "family": 4,
        "fashion_&_style": 5,
        "film_tv_&_video": 6,
        "fitness_&_health": 7,
        "food_&_dining": 8,
        "gaming": 9,
        "learning_&_educational": 10,
        "music": 11,
        "news_&_social_concern": 12,
        "other_hobbies": 13,
        "relationships": 14,
        "science_&_technology": 15,
        "sports": 16,
        "travel_&_adventure": 17,
        "youth_&_student_life": 18,
    }
    mapping_single_topic_overall = {
        "arts_&_culture": 0,
        "business_&_entrepreneurs": 1,
        "pop_culture": 2,
        "daily_life": 3,
        "sports_&_gaming": 4,
        "science_&_technology": 5,
    }

    if "sentiment_overall" in df.columns:
        df["sentiment_overall"] = df["sentiment_overall"].map(mapping_sentiment_overall)
    if "emo_overall" in df.columns:
        df["emo_overall"] = df["emo_overall"].map(mapping_emo_overall)
    if "topic_overall" in df.columns:
        df["topic_overall"] = df["topic_overall"].map(mapping_topic_overall)
    if "single_topic_overall" in df.columns:
        df["single_topic_overall"] = df["single_topic_overall"].map(mapping_single_topic_overall)

    return df


@torch.inference_mode()
def batched_transformer_predict(
    tokenizer,
    model,
    texts: List[str],
    batch_size: int = 64,
    max_length: int = 128,
    mode: str = "softmax",
) -> np.ndarray:
    outputs = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        enc = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}

        logits = model(**enc).logits.detach().cpu().numpy()

        if mode == "softmax":
            probs = softmax(logits, axis=1)
        elif mode == "sigmoid":
            probs = expit(logits)
        else:
            probs = logits

        outputs.append(probs)

    if not outputs:
        return np.empty((0, 0))

    return np.vstack(outputs)


def compute_basic_features(prep_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for row in prep_df.itertuples(index=False):
        text = row.full_text
        berkem_text = row.text_berkem

        try:
            text_len = len(berkem_text)
            word_count = len(str(berkem_text).split())
        except Exception:
            text_len = np.nan
            word_count = np.nan

        try:
            sentiment = sia.polarity_scores(berkem_text)
        except Exception:
            sentiment = {"neg": np.nan, "neu": np.nan, "pos": np.nan, "compound": np.nan}

        try:
            blob_sentiment = TextBlob(text).sentiment
            polarity = blob_sentiment.polarity
            subjectivity = blob_sentiment.subjectivity
        except Exception:
            polarity = np.nan
            subjectivity = np.nan

        try:
            readability_results = safe_readability(read_tokenized(berkem_text))
        except Exception:
            readability_results = None

        def rget(*keys):
            if readability_results is None:
                return np.nan
            try:
                val = readability_results
                for k in keys:
                    val = val[k]
                return val
            except Exception:
                return np.nan

        rows.append(
            {
                "full_text": text,
                "text_len": text_len,
                "word_count": word_count,
                "neg": sentiment["neg"],
                "neu": sentiment["neu"],
                "pos": sentiment["pos"],
                "compound": sentiment["compound"],
                "sentiment_overall": assign_sentiment(sentiment["neg"], sentiment["pos"])
                if not (pd.isna(sentiment["neg"]) or pd.isna(sentiment["pos"]))
                else np.nan,
                "grammar-word-score": np.nan,
                "grammar-sentence-score": np.nan,
                "subjectivity": subjectivity,
                "polarity": polarity,
                "Kincaid": rget("readability grades", "Kincaid"),
                "ARI": rget("readability grades", "ARI"),
                "Coleman-Liau": rget("readability grades", "Coleman-Liau"),
                "FleschReadingEase": rget("readability grades", "FleschReadingEase"),
                "GunningFogIndex": rget("readability grades", "GunningFogIndex"),
                "LIX": rget("readability grades", "LIX"),
                "SMOGIndex": rget("readability grades", "SMOGIndex"),
                "RIX": rget("readability grades", "RIX"),
                "DaleChallIndex": rget("readability grades", "DaleChallIndex"),
                "complex_words": rget("sentence info", "complex_words"),
                "complex_words_dc": rget("sentence info", "complex_words_dc"),
            }
        )

    return pd.DataFrame(rows)


def compute_pysentimiento_features(prep_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for row in prep_df.itertuples(index=False):
        text = row.full_text
        tweet_text = row.text_tweet

        emo_output = None
        hs_output = None

        try:
            emo_output = emotion_analyzer.predict(tweet_text)
        except Exception:
            pass

        try:
            hs_output = hate_speech_analyzer.predict(tweet_text)
        except Exception:
            pass

        rows.append(
            {
                "full_text": text,
                "emo_overall": getattr(emo_output, "output", np.nan),
                "emo_anger": getattr(emo_output, "probas", {}).get("anger", np.nan) if emo_output else np.nan,
                "emo_joy": getattr(emo_output, "probas", {}).get("joy", np.nan) if emo_output else np.nan,
                "emo_fear": getattr(emo_output, "probas", {}).get("fear", np.nan) if emo_output else np.nan,
                "emo_disgust": getattr(emo_output, "probas", {}).get("disgust", np.nan) if emo_output else np.nan,
                "emo_surprise": getattr(emo_output, "probas", {}).get("surprise", np.nan) if emo_output else np.nan,
                "emo_sadness": getattr(emo_output, "probas", {}).get("sadness", np.nan) if emo_output else np.nan,
                "emo_others": getattr(emo_output, "probas", {}).get("others", np.nan) if emo_output else np.nan,
                "hs_aggressive": getattr(hs_output, "probas", {}).get("aggressive", np.nan) if hs_output else np.nan,
                "hs_hateful": getattr(hs_output, "probas", {}).get("hateful", np.nan) if hs_output else np.nan,
                "hs_targeted": getattr(hs_output, "probas", {}).get("targeted", np.nan) if hs_output else np.nan,
                "hs_count": len(getattr(hs_output, "output", [])) if hs_output and hasattr(hs_output, "output") else np.nan,
            }
        )

    return pd.DataFrame(rows)


def compute_transformer_features(prep_df: pd.DataFrame, batch_size: int = 64) -> pd.DataFrame:
    texts = prep_df["full_text"].tolist()
    roberta_texts = prep_df["text_roberta"].tolist()

    irony_probs = batched_transformer_predict(
        tokenizer_irony,
        model_irony,
        roberta_texts,
        batch_size=batch_size,
        max_length=128,
        mode="softmax",
    )

    offensive_probs = batched_transformer_predict(
        tokenizer_offensive,
        model_offensive,
        roberta_texts,
        batch_size=batch_size,
        max_length=128,
        mode="softmax",
    )

    emoji_probs = batched_transformer_predict(
        tokenizer_emoji,
        model_emoji,
        roberta_texts,
        batch_size=batch_size,
        max_length=128,
        mode="softmax",
    )

    topic_probs = batched_transformer_predict(
        tokenizer_topic,
        model_topic,
        texts,
        batch_size=batch_size,
        max_length=128,
        mode="sigmoid",
    )

    topic_single_probs = batched_transformer_predict(
        tokenizer_topic_single,
        model_topic_single,
        roberta_texts,
        batch_size=batch_size,
        max_length=128,
        mode="sigmoid",
    )

    rows = []

    for i, text in enumerate(texts):
        row = {
            "full_text": text,
            "irony": int(irony_probs[i][1] >= 0.5) if irony_probs.shape[0] > i else np.nan,
            "offensive": offensive_probs[i][1] if offensive_probs.shape[0] > i else np.nan,
            "emoji": int(np.argmax(emoji_probs[i])) if emoji_probs.shape[0] > i else np.nan,
        }

        if topic_probs.shape[0] > i:
            row["topic_count"] = int(np.sum((topic_probs[i] >= 0.5).astype(int)))
            row["topic_overall"] = class_mapping[int(np.argmax(topic_probs[i]))]
            for j, label in enumerate(class_mapping):
                row[label] = topic_probs[i][j]
        else:
            row["topic_count"] = np.nan
            row["topic_overall"] = np.nan
            for label in class_mapping:
                row[label] = np.nan

        if topic_single_probs.shape[0] > i:
            row["single_topic_count"] = int(np.sum((topic_single_probs[i] >= 0.5).astype(int)))
            row["single_topic_overall"] = class_mapping_single[int(np.argmax(topic_single_probs[i]))]
            for j, label in enumerate(class_mapping_single):
                row["single_" + label] = topic_single_probs[i][j]
        else:
            row["single_topic_count"] = np.nan
            row["single_topic_overall"] = np.nan
            for label in class_mapping_single:
                row["single_" + label] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def add_all_m_features(
    df: pd.DataFrame,
    text_col: str = "full_text",
    batch_size: int = 64,
) -> pd.DataFrame:
    start_time = time.perf_counter()

    df = df.copy()
    df = df.dropna(subset=[text_col]).copy()
    df[text_col] = df[text_col].astype(str)

    if text_col != "full_text":
        df = df.rename(columns={text_col: "full_text"})
        rename_back = True
    else:
        rename_back = False

    unique_texts = df["full_text"].drop_duplicates().tolist()

    print(f"Preparing {len(unique_texts)} unique texts...")
    prep_df = build_preprocessed_texts(unique_texts)

    print("Computing basic features...")
    basic_df = compute_basic_features(prep_df)

    print("Computing pysentimiento features...")
    pysent_df = compute_pysentimiento_features(prep_df)

    print(f"Computing transformer features in batches of {batch_size}...")
    transformer_df = compute_transformer_features(prep_df, batch_size=batch_size)

    feat_df = basic_df.merge(pysent_df, on="full_text", how="left")
    feat_df = feat_df.merge(transformer_df, on="full_text", how="left")

    feat_df = m_mapping_category_values(feat_df)

    df = df.merge(feat_df, on="full_text", how="left")

    helper_cols = [c for c in ["text_berkem", "text_grammar", "text_roberta", "text_tweet"] if c in df.columns]
    if helper_cols:
        df = df.drop(columns=helper_cols)

    if rename_back:
        df = df.rename(columns={"full_text": text_col})

    gc.collect()

    finish_time = time.perf_counter()
    print(f"processed {len(df)} rows")
    print(f"finished in {(finish_time - start_time) / 60:.2f} minutes")

    return df