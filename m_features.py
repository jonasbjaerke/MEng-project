import re
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

from pysentimiento.preprocessing import preprocess_tweet
from pysentimiento import create_analyzer

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax, expit

import readability
import syntok.segmenter as segmenter


# ==========================================================
# LOAD MODELS ONCE
# ==========================================================

sia = SentimentIntensityAnalyzer()

emotion_analyzer = create_analyzer(task="emotion", lang="en")
hate_speech_analyzer = create_analyzer(task="hate_speech", lang="en")

MODEL_IRONY = "cardiffnlp/twitter-roberta-base-irony"
MODEL_OFFENSIVE = "cardiffnlp/twitter-roberta-base-offensive"
MODEL_EMOJI = "cardiffnlp/twitter-roberta-base-emoji"
MODEL_TOPIC_MULTI = "cardiffnlp/tweet-topic-21-multi"
MODEL_TOPIC_SINGLE = "cardiffnlp/tweet-topic-21-single"

tokenizer_irony = AutoTokenizer.from_pretrained(MODEL_IRONY)
model_irony = AutoModelForSequenceClassification.from_pretrained(MODEL_IRONY)

tokenizer_offensive = AutoTokenizer.from_pretrained(MODEL_OFFENSIVE)
model_offensive = AutoModelForSequenceClassification.from_pretrained(MODEL_OFFENSIVE)

tokenizer_emoji = AutoTokenizer.from_pretrained(MODEL_EMOJI)
model_emoji = AutoModelForSequenceClassification.from_pretrained(MODEL_EMOJI)

tokenizer_topic = AutoTokenizer.from_pretrained(MODEL_TOPIC_MULTI)
model_topic = AutoModelForSequenceClassification.from_pretrained(MODEL_TOPIC_MULTI)
class_mapping = model_topic.config.id2label

tokenizer_topic_single = AutoTokenizer.from_pretrained(MODEL_TOPIC_SINGLE)
model_topic_single = AutoModelForSequenceClassification.from_pretrained(MODEL_TOPIC_SINGLE)
class_mapping_single = model_topic_single.config.id2label

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for m in [model_irony, model_offensive, model_emoji, model_topic, model_topic_single]:
    m.to(DEVICE)
    m.eval()


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
    new_text = []
    for t in text.split(" "):
        t = "@user" if t.startswith("@") else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)

def read_tokenized(text):
    return "\n\n".join(
        "\n".join(
            " ".join(token.value for token in sentence)
            for sentence in paragraph
        )
        for paragraph in segmenter.analyze(text)
    )


# ==========================================================
# TRANSFORMER BATCH FORWARD
# ==========================================================

@torch.no_grad()
def batch_forward(tokenizer, model, texts, batch_size=64):
    outputs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        logits = model(**enc).logits.detach().cpu().numpy()
        outputs.append(logits)
    return np.vstack(outputs)


# ==========================================================
# MAIN FEATURE FUNCTION
# ==========================================================

def add_all_m_features(df, batch_size=64):

    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)

    unique_texts = df["text"].unique().tolist()

    text_len = {}
    word_count = {}
    sentiment_scores = {}
    subjectivity_score = {}
    polarity_score = {}
    emo_results = {}
    hs_results = {}
    irony_scores = {}
    offensive_scores = {}
    emoji_top = {}
    readability_results = {}
    topic_scores = {}
    topic_pred = {}
    topic_scores_single = {}
    topic_pred_single = {}

    # ---------------------------
    # Cheap per-text features
    # ---------------------------
    for t in unique_texts:
        clean = berkem_preprocess(t)
        text_len[t] = len(clean)
        word_count[t] = len(clean.split())
        sentiment_scores[t] = sia.polarity_scores(clean)
        tb = TextBlob(t).sentiment
        polarity_score[t] = tb.polarity
        subjectivity_score[t] = tb.subjectivity
        try:
            readability_results[t] = readability.getmeasures(read_tokenized(clean), lang="en")
        except:
            readability_results[t] = None

    # ---------------------------
    # Emotion + Hate
    # ---------------------------
    prep = [preprocess_tweet(t) for t in unique_texts]
    emo_preds = emotion_analyzer.predict(prep)
    hs_preds = hate_speech_analyzer.predict(prep)

    for t, e, h in zip(unique_texts, emo_preds, hs_preds):
        emo_results[t] = e
        hs_results[t] = h

    # ---------------------------
    # RoBERTa models
    # ---------------------------
    rob_texts = [roberta_preprocess(t) for t in unique_texts]

    irony_probs = softmax(batch_forward(tokenizer_irony, model_irony, rob_texts, batch_size), axis=1)
    offensive_probs = softmax(batch_forward(tokenizer_offensive, model_offensive, rob_texts, batch_size), axis=1)
    emoji_probs = softmax(batch_forward(tokenizer_emoji, model_emoji, rob_texts, batch_size), axis=1)
    topic_probs = expit(batch_forward(tokenizer_topic, model_topic, rob_texts, batch_size))
    topic_single_probs = expit(batch_forward(tokenizer_topic_single, model_topic_single, rob_texts, batch_size))

    for i, t in enumerate(unique_texts):
        irony_scores[t] = irony_probs[i]
        offensive_scores[t] = offensive_probs[i]
        emoji_top[t] = int(np.argmax(emoji_probs[i]))
        topic_scores[t] = topic_probs[i]
        topic_pred[t] = (topic_probs[i] >= 0.5).astype(int)
        topic_scores_single[t] = topic_single_probs[i]
        topic_pred_single[t] = (topic_single_probs[i] >= 0.5).astype(int)

    # ==========================================================
    # Attach to DataFrame (exact columns from your notebook)
    # ==========================================================

    df["text_len"] = df["text"].map(text_len)
    df["word_count"] = df["text"].map(word_count)

    df["neg"] = df["text"].map(lambda x: sentiment_scores[x]["neg"])
    df["neu"] = df["text"].map(lambda x: sentiment_scores[x]["neu"])
    df["pos"] = df["text"].map(lambda x: sentiment_scores[x]["pos"])
    df["compound"] = df["text"].map(lambda x: sentiment_scores[x]["compound"])

    df["subjectivity"] = df["text"].map(subjectivity_score)
    df["polarity"] = df["text"].map(polarity_score)

    df["emo_overall"] = df["text"].map(lambda x: emo_results[x].output)
    df["emo_anger"] = df["text"].map(lambda x: emo_results[x].probas.get("anger", 0))
    df["emo_joy"] = df["text"].map(lambda x: emo_results[x].probas.get("joy", 0))
    df["emo_fear"] = df["text"].map(lambda x: emo_results[x].probas.get("fear", 0))
    df["emo_disgust"] = df["text"].map(lambda x: emo_results[x].probas.get("disgust", 0))
    df["emo_surprise"] = df["text"].map(lambda x: emo_results[x].probas.get("surprise", 0))
    df["emo_sadness"] = df["text"].map(lambda x: emo_results[x].probas.get("sadness", 0))
    df["emo_others"] = df["text"].map(lambda x: emo_results[x].probas.get("others", 0))

    df["hs_aggressive"] = df["text"].map(lambda x: hs_results[x].probas.get("aggressive", 0))
    df["hs_hateful"] = df["text"].map(lambda x: hs_results[x].probas.get("hateful", 0))
    df["hs_targeted"] = df["text"].map(lambda x: hs_results[x].probas.get("targeted", 0))
    df["hs_count"] = df["text"].map(lambda x: len(hs_results[x].output))

    df["irony"] = df["text"].map(lambda x: 0 if irony_scores[x][1] < 0.5 else 1)
    df["offensive"] = df["text"].map(lambda x: offensive_scores[x][1])
    df["emoji"] = df["text"].map(emoji_top)

    labels_multi = [class_mapping[i] for i in range(len(class_mapping))]
    for i, lab in enumerate(labels_multi):
        df[lab] = df["text"].map(lambda x: topic_scores[x][i])

    df["topic_count"] = df["text"].map(lambda x: int(np.sum(topic_pred[x])))
    df["topic_overall"] = df["text"].map(lambda x: labels_multi[int(np.argmax(topic_scores[x]))])

    labels_single = [class_mapping_single[i] for i in range(len(class_mapping_single))]
    for i, lab in enumerate(labels_single):
        df["single_" + lab] = df["text"].map(lambda x: topic_scores_single[x][i])

    df["single_topic_count"] = df["text"].map(lambda x: int(np.sum(topic_pred_single[x])))
    df["single_topic_overall"] = df["text"].map(lambda x: labels_single[int(np.argmax(topic_scores_single[x]))])

    return df
