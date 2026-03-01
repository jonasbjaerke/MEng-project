import os
import gc
import re
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Silence HF noise
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers.utils import logging
logging.set_verbosity_error()

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from pysentimiento import create_analyzer


def status(msg):
    print(f"\r{msg}", end="", flush=True)

# ==========================================================
# DEVICE SETUP
# ==========================================================

torch.set_float32_matmul_precision("high")

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("Using device:", DEVICE)


# ==========================================================
# MODEL LOADER
# ==========================================================

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


# Cardiff models
tokenizer_irony, model_irony = load_model("cardiffnlp/twitter-roberta-base-irony")
tokenizer_offensive, model_offensive = load_model("cardiffnlp/twitter-roberta-base-offensive")
tokenizer_emoji, model_emoji = load_model("cardiffnlp/twitter-roberta-base-emoji")
tokenizer_topic, model_topic = load_model("cardiffnlp/tweet-topic-21-multi")
tokenizer_topic_single, model_topic_single = load_model("cardiffnlp/tweet-topic-21-single")

# Load pysentimiento ONLY to extract raw model + tokenizer
emotion_analyzer = create_analyzer(task="emotion", lang="en", device=DEVICE)
hate_speech_analyzer = create_analyzer(task="hate_speech", lang="en", device=DEVICE)

tokenizer_emotion = emotion_analyzer.tokenizer
model_emotion = emotion_analyzer.model

tokenizer_hate = hate_speech_analyzer.tokenizer
model_hate = hate_speech_analyzer.model

model_emotion.eval()
model_hate.eval()

class_mapping = list(model_topic.config.id2label.values())
class_mapping_single = list(model_topic_single.config.id2label.values())

emotion_labels = ["anger", "joy", "fear", "disgust", "surprise", "sadness", "others"]
hate_labels = ["hateful", "aggressive", "targeted"]

sia = SentimentIntensityAnalyzer()


# ==========================================================
# PREPROCESSING
# ==========================================================

def berkem_preprocess(text):
    text = re.sub(r"RT @\w+: ", " ", text)
    text = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+://\S+)", " ", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"&amp", "", text)
    return text.lower().strip()

def roberta_preprocess(text):
    return " ".join(
        "@user" if t.startswith("@") else
        "http" if t.startswith("http") else t
        for t in text.split()
    )


# ==========================================================
# FAST BATCH FORWARD
# ==========================================================

@torch.no_grad()
def batch_forward(tokenizer, model, texts, batch_size=32, activation="softmax"):

    outputs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        logits = model(**enc).logits

        if activation == "softmax":
            probs = torch.softmax(logits, dim=1)
        elif activation == "sigmoid":
            probs = torch.sigmoid(logits)
        else:
            probs = logits

        outputs.append(probs.cpu().numpy())

    return np.vstack(outputs)


# ==========================================================
# MAIN FEATURE FUNCTION
# ==========================================================

def add_all_m_features(df, batch_size=64):

    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)

    unique_texts = df["text"].unique().tolist()
    #print("Unique texts:", len(unique_texts))

    clean_map = {t: berkem_preprocess(t) for t in unique_texts}
    rob_map = {t: roberta_preprocess(t) for t in unique_texts}

    # Basic features
    text_len = {t: len(clean_map[t]) for t in unique_texts}
    word_count = {t: len(clean_map[t].split()) for t in unique_texts}
    sentiment_scores = {t: sia.polarity_scores(clean_map[t]) for t in unique_texts}
    tb_scores = {t: TextBlob(t).sentiment for t in unique_texts}

    rob_texts = [rob_map[t] for t in unique_texts]

    # Transformer passes
  #  status("Running emotion model...")
    emotion_probs = batch_forward(tokenizer_emotion, model_emotion, rob_texts, batch_size)

   # status("Running hate model...")
    hate_probs = batch_forward(tokenizer_hate, model_hate, rob_texts, batch_size)
   # status("Running irony model...")
    irony_probs = batch_forward(tokenizer_irony, model_irony, rob_texts, batch_size)
   # status("Running offensive model...")
    offensive_probs = batch_forward(tokenizer_offensive, model_offensive, rob_texts, batch_size)
   # status("Running emoji model...")
    emoji_probs = batch_forward(tokenizer_emoji, model_emoji, rob_texts, batch_size)
   # status("Running topic model...")
    topic_probs = batch_forward(tokenizer_topic, model_topic, rob_texts, batch_size, activation="sigmoid")
    #status("Running topic-singel model...")
    topic_single_probs = batch_forward(tokenizer_topic_single, model_topic_single, rob_texts, batch_size, activation="sigmoid")

    mapping_index = {t: i for i, t in enumerate(unique_texts)}
    def idx(x): return mapping_index[x]

    # Attach features
    df["text_len"] = df["text"].map(text_len)
    df["word_count"] = df["text"].map(word_count)

    df["neg"] = df["text"].map(lambda x: sentiment_scores[x]["neg"])
    df["neu"] = df["text"].map(lambda x: sentiment_scores[x]["neu"])
    df["pos"] = df["text"].map(lambda x: sentiment_scores[x]["pos"])
    df["compound"] = df["text"].map(lambda x: sentiment_scores[x]["compound"])

    df["subjectivity"] = df["text"].map(lambda x: tb_scores[x].subjectivity)
    df["polarity"] = df["text"].map(lambda x: tb_scores[x].polarity)

    # Emotion
    df["emo_overall"] = df["text"].map(lambda x: emotion_labels[int(np.argmax(emotion_probs[idx(x)]))])
    for i, lab in enumerate(emotion_labels):
        df[f"emo_{lab}"] = df["text"].map(lambda x: emotion_probs[idx(x)][i])

    # Hate
    for i, lab in enumerate(hate_labels):
        df[f"hs_{lab}"] = df["text"].map(lambda x: hate_probs[idx(x)][i])
    df["hs_count"] = df["text"].map(lambda x: int(np.sum(hate_probs[idx(x)] > 0.5)))

    # Irony / Offensive / Emoji
    df["irony"] = df["text"].map(lambda x: int(irony_probs[idx(x)][1] >= 0.5))
    df["offensive"] = df["text"].map(lambda x: offensive_probs[idx(x)][1])
    df["emoji"] = df["text"].map(lambda x: int(np.argmax(emoji_probs[idx(x)])))

    # Topics multi
    for i, lab in enumerate(class_mapping):
        df[lab] = df["text"].map(lambda x: topic_probs[idx(x)][i])
    df["topic_count"] = df["text"].map(lambda x: int(np.sum(topic_probs[idx(x)] >= 0.5)))
    df["topic_overall"] = df["text"].map(lambda x: class_mapping[int(np.argmax(topic_probs[idx(x)]))])

    # Topics single
    for i, lab in enumerate(class_mapping_single):
        df["single_" + lab] = df["text"].map(lambda x: topic_single_probs[idx(x)][i])
    df["single_topic_count"] = df["text"].map(lambda x: int(np.sum(topic_single_probs[idx(x)] >= 0.5)))
    df["single_topic_overall"] = df["text"].map(
        lambda x: class_mapping_single[int(np.argmax(topic_single_probs[idx(x)]))]
    )

    return df
