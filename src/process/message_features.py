# # ==========================================================
# # IMPORTS
# # ==========================================================

# import os
# import gc
# import re
# import numpy as np
# import pandas as pd

# # Silence HF noise
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# from transformers.utils import logging
# logging.set_verbosity_error()

# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from nltk.sentiment import SentimentIntensityAnalyzer
# from textblob import TextBlob
# from pysentimiento import create_analyzer


# # ==========================================================
# # DEVICE SETUP
# # ==========================================================

# torch.set_float32_matmul_precision("high")

# if torch.backends.mps.is_available():
#     DEVICE = torch.device("mps")
# elif torch.cuda.is_available():
#     DEVICE = torch.device("cuda")
# else:
#     DEVICE = torch.device("cpu")

# print("Using device:", DEVICE)


# # ==========================================================
# # MODEL LOADER
# # ==========================================================

# def load_model(model_name):
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSequenceClassification.from_pretrained(model_name)

#     model.to(DEVICE)
#     model.eval()

#     return tokenizer, model


# # ==========================================================
# # LOAD MODELS
# # ==========================================================

# tokenizer_irony, model_irony = load_model("cardiffnlp/twitter-roberta-base-irony")
# tokenizer_offensive, model_offensive = load_model("cardiffnlp/twitter-roberta-base-offensive")
# tokenizer_emoji, model_emoji = load_model("cardiffnlp/twitter-roberta-base-emoji")
# tokenizer_topic, model_topic = load_model("cardiffnlp/tweet-topic-21-multi")
# tokenizer_topic_single, model_topic_single = load_model("cardiffnlp/tweet-topic-21-single")

# emotion_analyzer = create_analyzer(task="emotion", lang="en", device=DEVICE)
# hate_speech_analyzer = create_analyzer(task="hate_speech", lang="en", device=DEVICE)

# tokenizer_emotion = emotion_analyzer.tokenizer
# model_emotion = emotion_analyzer.model
# tokenizer_hate = hate_speech_analyzer.tokenizer
# model_hate = hate_speech_analyzer.model

# model_emotion.eval()
# model_hate.eval()

# class_mapping = list(model_topic.config.id2label.values())
# class_mapping_single = list(model_topic_single.config.id2label.values())

# emotion_labels = ["anger", "joy", "fear", "disgust", "surprise", "sadness", "others"]
# hate_labels = ["hateful", "aggressive", "targeted"]

# sia = SentimentIntensityAnalyzer()


# # ==========================================================
# # PREPROCESSING
# # ==========================================================

# def berkem_preprocess(text):
#     text = re.sub(r"RT @\w+: ", " ", text)
#     text = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+://\S+)", " ", text)
#     text = re.sub(r"#", "", text)
#     text = re.sub(r"&amp", "", text)
#     return text.lower().strip()

# def roberta_preprocess(text):
#     return " ".join(
#         "@user" if t.startswith("@") else
#         "http" if t.startswith("http") else t
#         for t in text.split()
#     )


# # ==========================================================
# # FAST BATCH FORWARD
# # ==========================================================

# @torch.no_grad()
# def batch_forward(tokenizer, model, texts, batch_size=128, activation="softmax"):

#     outputs = []

#     for i in range(0, len(texts), batch_size):
#         batch = texts[i:i+batch_size]

#         enc = tokenizer(
#             batch,
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#             max_length=128  # original length
#         )

#         enc = {k: v.to(DEVICE) for k, v in enc.items()}
#         logits = model(**enc).logits

#         if activation == "softmax":
#             probs = torch.softmax(logits, dim=1)
#         elif activation == "sigmoid":
#             probs = torch.sigmoid(logits)
#         else:
#             probs = logits

#         outputs.append(probs.cpu().numpy())

#     return np.vstack(outputs)


# # ==========================================================
# # MAIN FEATURE FUNCTION
# # ==========================================================

# def add_all_m_features(df, batch_size=128):

#     df = df.copy()
#     df["text"] = df["text"].fillna("").astype(str)

#     unique_texts = df["text"].unique().tolist()

#     clean_map = {t: berkem_preprocess(t) for t in unique_texts}
#     rob_map = {t: roberta_preprocess(t) for t in unique_texts}

#     # Basic features
#     text_len = {t: len(clean_map[t]) for t in unique_texts}
#     word_count = {t: len(clean_map[t].split()) for t in unique_texts}
#     sentiment_scores = {t: sia.polarity_scores(clean_map[t]) for t in unique_texts}
#     tb_scores = {t: TextBlob(t).sentiment for t in unique_texts}

#     rob_texts = [rob_map[t] for t in unique_texts]

#     # Transformer passes
#     emotion_probs = batch_forward(tokenizer_emotion, model_emotion, rob_texts, batch_size)
#     hate_probs = batch_forward(tokenizer_hate, model_hate, rob_texts, batch_size)
#     irony_probs = batch_forward(tokenizer_irony, model_irony, rob_texts, batch_size)
#     offensive_probs = batch_forward(tokenizer_offensive, model_offensive, rob_texts, batch_size)
#     emoji_probs = batch_forward(tokenizer_emoji, model_emoji, rob_texts, batch_size)
#     topic_probs = batch_forward(tokenizer_topic, model_topic, rob_texts, batch_size, activation="sigmoid")
#     topic_single_probs = batch_forward(tokenizer_topic_single, model_topic_single, rob_texts, batch_size, activation="sigmoid")

#     mapping_index = {t: i for i, t in enumerate(unique_texts)}
#     def idx(x): return mapping_index[x]

#     # Attach features
#     df["text_len"] = df["text"].map(text_len)
#     df["word_count"] = df["text"].map(word_count)

#     df["neg"] = df["text"].map(lambda x: sentiment_scores[x]["neg"])
#     df["neu"] = df["text"].map(lambda x: sentiment_scores[x]["neu"])
#     df["pos"] = df["text"].map(lambda x: sentiment_scores[x]["pos"])
#     df["compound"] = df["text"].map(lambda x: sentiment_scores[x]["compound"])

#     df["subjectivity"] = df["text"].map(lambda x: tb_scores[x].subjectivity)
#     df["polarity"] = df["text"].map(lambda x: tb_scores[x].polarity)

#     # Emotion
#     df["emo_overall"] = df["text"].map(lambda x: emotion_labels[int(np.argmax(emotion_probs[idx(x)]))])
#     for i, lab in enumerate(emotion_labels):
#         df[f"emo_{lab}"] = df["text"].map(lambda x: emotion_probs[idx(x)][i])

#     # Hate
#     for i, lab in enumerate(hate_labels):
#         df[f"hs_{lab}"] = df["text"].map(lambda x: hate_probs[idx(x)][i])
#     df["hs_count"] = df["text"].map(lambda x: int(np.sum(hate_probs[idx(x)] > 0.5)))

#     # Irony / Offensive / Emoji
#     df["irony"] = df["text"].map(lambda x: int(irony_probs[idx(x)][1] >= 0.5))
#     df["offensive"] = df["text"].map(lambda x: offensive_probs[idx(x)][1])
#     df["emoji"] = df["text"].map(lambda x: int(np.argmax(emoji_probs[idx(x)])))

#     # Topics multi
#     for i, lab in enumerate(class_mapping):
#         df[lab] = df["text"].map(lambda x: topic_probs[idx(x)][i])
#     df["topic_count"] = df["text"].map(lambda x: int(np.sum(topic_probs[idx(x)] >= 0.5)))
#     df["topic_overall"] = df["text"].map(lambda x: class_mapping[int(np.argmax(topic_probs[idx(x)]))])

#     # Topics single
#     for i, lab in enumerate(class_mapping_single):
#         df["single_" + lab] = df["text"].map(lambda x: topic_single_probs[idx(x)][i])
#     df["single_topic_count"] = df["text"].map(lambda x: int(np.sum(topic_single_probs[idx(x)] >= 0.5)))
#     df["single_topic_overall"] = df["text"].map(
#         lambda x: class_mapping_single[int(np.argmax(topic_single_probs[idx(x)]))]
#     )

#     gc.collect()
#     return df

import os
import gc
import re
from pathlib import Path

import numpy as np
import pandas as pd
import nltk
import readability

try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except Exception:
    textstat = None
    TEXTSTAT_AVAILABLE = False

try:
    import preprocessor as tweet_preprocessor
    PREPROCESSOR_AVAILABLE = True
except Exception:
    tweet_preprocessor = None
    PREPROCESSOR_AVAILABLE = False

try:
    import syntok.segmenter as segmenter
    SYNTOK_AVAILABLE = True
except Exception:
    segmenter = None
    SYNTOK_AVAILABLE = False

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers.utils import logging
logging.set_verbosity_error()

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from pysentimiento import create_analyzer
from pysentimiento.preprocessing import preprocess_tweet


def ensure_nltk_resources():
    resources = [
        ("sentiment/vader_lexicon.zip", "vader_lexicon"),
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
    ]

    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name)


ensure_nltk_resources()

torch.set_float32_matmul_precision("high")

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("Using device:", DEVICE)


def load_model(model_name):
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

grammar_check_tool = None
GRAMMAR_AVAILABLE = True


def get_grammar_tool():
    global grammar_check_tool, GRAMMAR_AVAILABLE

    if not GRAMMAR_AVAILABLE:
        return None

    if grammar_check_tool is None:
        try:
            import language_tool_python
            grammar_check_tool = language_tool_python.LanguageTool("en-US")
        except Exception:
            GRAMMAR_AVAILABLE = False
            return None

    return grammar_check_tool


def berkem_preprocess(text):
    text = str(text)
    text = re.sub(r"RT @\w+: ", " ", text)
    text = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text)
    if PREPROCESSOR_AVAILABLE:
        tweet_preprocessor.set_options(
            tweet_preprocessor.OPT.URL,
            tweet_preprocessor.OPT.EMOJI,
            tweet_preprocessor.OPT.MENTION,
            tweet_preprocessor.OPT.NUMBER,
        )
        text = tweet_preprocessor.clean(text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"&amp", "", text)
    text = text.lower()
    return text


def grammar_preprocess(text):
    text = str(text)
    text = re.sub(r"RT @\w+: ", " ", text)
    text = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text)
    if PREPROCESSOR_AVAILABLE:
        tweet_preprocessor.set_options(
            tweet_preprocessor.OPT.URL,
            tweet_preprocessor.OPT.EMOJI,
            tweet_preprocessor.OPT.MENTION,
            tweet_preprocessor.OPT.NUMBER,
        )
        text = tweet_preprocessor.clean(text)
    return text


def roberta_preprocess(text):
    new_text = []
    for t in str(text).split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)


def assign_sentiment(neg, pos):
    if neg > pos:
        return "negative"
    elif pos > neg:
        return "positive"
    return "neutral"


def read_tokenized(text):
    text = str(text)
    if SYNTOK_AVAILABLE:
        return "\n\n".join(
            "\n".join(" ".join(token.value for token in sentence) for sentence in paragraph)
            for paragraph in segmenter.analyze(text)
        )
    try:
        sentences = nltk.sent_tokenize(text)
        return "\n".join(sentences)
    except Exception:
        return text


def empty_readability_scores():
    return {
        "Kincaid": np.nan,
        "ARI": np.nan,
        "Coleman-Liau": np.nan,
        "FleschReadingEase": np.nan,
        "GunningFogIndex": np.nan,
        "LIX": np.nan,
        "SMOGIndex": np.nan,
        "RIX": np.nan,
        "DaleChallIndex": np.nan,
        "complex_words": np.nan,
        "complex_words_dc": np.nan,
    }


def get_readability_scores(berkem_text):
    berkem_text = str(berkem_text)
    try:
        result = readability.getmeasures(read_tokenized(berkem_text), lang="en")
        grades = result.get("readability grades", {})
        sent_info = result.get("sentence info", {})
        return {
            "Kincaid": grades.get("Kincaid", np.nan),
            "ARI": grades.get("ARI", np.nan),
            "Coleman-Liau": grades.get("Coleman-Liau", np.nan),
            "FleschReadingEase": grades.get("FleschReadingEase", np.nan),
            "GunningFogIndex": grades.get("GunningFogIndex", np.nan),
            "LIX": grades.get("LIX", np.nan),
            "SMOGIndex": grades.get("SMOGIndex", np.nan),
            "RIX": grades.get("RIX", np.nan),
            "DaleChallIndex": grades.get("DaleChallIndex", np.nan),
            "complex_words": sent_info.get("complex_words", np.nan),
            "complex_words_dc": sent_info.get("complex_words_dc", np.nan),
        }
    except Exception:
        fallback = empty_readability_scores()
        if TEXTSTAT_AVAILABLE and berkem_text.strip():
            try:
                fallback.update({
                    "Kincaid": textstat.flesch_kincaid_grade(berkem_text),
                    "ARI": textstat.automated_readability_index(berkem_text),
                    "Coleman-Liau": textstat.coleman_liau_index(berkem_text),
                    "FleschReadingEase": textstat.flesch_reading_ease(berkem_text),
                    "GunningFogIndex": textstat.gunning_fog(berkem_text),
                    "SMOGIndex": textstat.smog_index(berkem_text),
                    "DaleChallIndex": textstat.dale_chall_readability_score(berkem_text),
                })
            except Exception:
                pass
        return fallback


def get_grammar_score(text):
    tool = get_grammar_tool()
    if tool is None:
        return [np.nan, np.nan]

    sentences = nltk.tokenize.sent_tokenize(text)
    if len(sentences) == 0:
        return [np.nan, np.nan]

    scores_word_based_sentence = []
    scores_sentence_based_sentence = []

    for sentence in sentences:
        matches = tool.check(sentence)
        count_errors = len(matches)
        scores_sentence_based_sentence.append(min(count_errors, 1))
        scores_word_based_sentence.append(count_errors)

    word_count = len(nltk.tokenize.word_tokenize(text))
    score_word_based = np.nan if word_count == 0 else 1 - (np.sum(scores_word_based_sentence) / word_count)
    sentence_count = len(sentences)
    score_sentence_based = 1 - np.sum(np.sum(scores_sentence_based_sentence) / sentence_count)
    return [score_word_based, score_sentence_based]


@torch.no_grad()
def batch_forward(tokenizer, model, texts, batch_size=128, activation="softmax"):
    outputs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
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


def m_mapping_category_values(df):
    df = df.copy()

    mapping_sentiment_overall = {"neutral": 0, "negative": 1, "positive": 2}
    mapping_emo_overall = {"others": 0, "joy": 1, "sadness": 2, "disgust": 3, "surprise": 4, "anger": 5, "fear": 6}
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

    df["sentiment_overall"] = df["sentiment_overall"].map(mapping_sentiment_overall)
    df["emo_overall"] = df["emo_overall"].map(mapping_emo_overall)
    df["topic_overall"] = df["topic_overall"].map(mapping_topic_overall)
    df["single_topic_overall"] = df["single_topic_overall"].map(mapping_single_topic_overall)
    return df


def add_all_m_features(df, batch_size=128):
    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)

    unique_texts = df["text"].unique().tolist()

    text_len = {}
    word_count = {}
    sentiment_scores = {}
    grammar_score = {}
    subjectivity_score = {}
    polarity_score = {}
    emo_results = {}
    hs_results = {}
    readability_results = {}

    clean_map = {}
    grammar_map = {}
    rob_map = {}

    for text in unique_texts:
        grammar_text = grammar_preprocess(text)
        berkem_text = berkem_preprocess(text)
        clean_map[text] = berkem_text
        grammar_map[text] = grammar_text
        rob_map[text] = roberta_preprocess(text)

        text_len[text] = len(berkem_text)
        word_count[text] = len(str(berkem_text).split())
        sentiment_scores[text] = sia.polarity_scores(berkem_text)
        grammar_score[text] = get_grammar_score(grammar_text)
        subjectivity_score[text] = TextBlob(text).sentiment[1]
        polarity_score[text] = TextBlob(text).sentiment[0]
        emo_results[text] = emotion_analyzer.predict(preprocess_tweet(text))
        hs_results[text] = hate_speech_analyzer.predict(preprocess_tweet(text))
        readability_results[text] = get_readability_scores(berkem_text)

    rob_texts = [rob_map[t] for t in unique_texts]
    raw_texts = unique_texts

    irony_probs = batch_forward(tokenizer_irony, model_irony, rob_texts, batch_size, activation="softmax")
    offensive_probs = batch_forward(tokenizer_offensive, model_offensive, rob_texts, batch_size, activation="softmax")
    emoji_probs = batch_forward(tokenizer_emoji, model_emoji, rob_texts, batch_size, activation="softmax")
    topic_probs = batch_forward(tokenizer_topic, model_topic, raw_texts, batch_size, activation="sigmoid")
    topic_single_probs = batch_forward(tokenizer_topic_single, model_topic_single, rob_texts, batch_size, activation="sigmoid")

    mapping_index = {t: i for i, t in enumerate(unique_texts)}

    def idx(x):
        return mapping_index[x]

    df["text_len"] = df["text"].map(lambda x: text_len[str(x)])
    df["word_count"] = df["text"].map(lambda x: word_count[str(x)])

    df["neg"] = df["text"].map(lambda x: sentiment_scores[x]["neg"])
    df["neu"] = df["text"].map(lambda x: sentiment_scores[x]["neu"])
    df["pos"] = df["text"].map(lambda x: sentiment_scores[x]["pos"])
    df["compound"] = df["text"].map(lambda x: sentiment_scores[x]["compound"])
    df["sentiment_overall"] = df["text"].map(
        lambda x: assign_sentiment(sentiment_scores[x]["neg"], sentiment_scores[x]["pos"])
    )

    df["grammar-word-score"] = df["text"].map(lambda x: grammar_score[x][0])
    df["grammar-sentence-score"] = df["text"].map(lambda x: grammar_score[x][1])

    df["subjectivity"] = df["text"].map(lambda x: subjectivity_score[x])
    df["polarity"] = df["text"].map(lambda x: polarity_score[x])

    df["emo_overall"] = df["text"].map(lambda x: emo_results[x].output)
    df["emo_anger"] = df["text"].map(lambda x: emo_results[x].probas["anger"])
    df["emo_joy"] = df["text"].map(lambda x: emo_results[x].probas["joy"])
    df["emo_fear"] = df["text"].map(lambda x: emo_results[x].probas["fear"])
    df["emo_disgust"] = df["text"].map(lambda x: emo_results[x].probas["disgust"])
    df["emo_surprise"] = df["text"].map(lambda x: emo_results[x].probas["surprise"])
    df["emo_sadness"] = df["text"].map(lambda x: emo_results[x].probas["sadness"])
    df["emo_others"] = df["text"].map(lambda x: emo_results[x].probas["others"])

    df["hs_aggressive"] = df["text"].map(lambda x: hs_results[x].probas["aggressive"])
    df["hs_hateful"] = df["text"].map(lambda x: hs_results[x].probas["hateful"])
    df["hs_targeted"] = df["text"].map(lambda x: hs_results[x].probas["targeted"])
    df["hs_count"] = df["text"].map(lambda x: len(hs_results[x].output))

    df["irony"] = df["text"].map(lambda x: 0 if irony_probs[idx(x)][1] < 0.5 else 1)
    df["offensive"] = df["text"].map(lambda x: offensive_probs[idx(x)][1])

    df["emoji"] = df["text"].map(lambda x: int(np.argsort(emoji_probs[idx(x)])[::-1][0]))

    df["Kincaid"] = df["text"].map(lambda x: readability_results[x]["Kincaid"])
    df["ARI"] = df["text"].map(lambda x: readability_results[x]["ARI"])
    df["Coleman-Liau"] = df["text"].map(lambda x: readability_results[x]["Coleman-Liau"])
    df["FleschReadingEase"] = df["text"].map(lambda x: readability_results[x]["FleschReadingEase"])
    df["GunningFogIndex"] = df["text"].map(lambda x: readability_results[x]["GunningFogIndex"])
    df["LIX"] = df["text"].map(lambda x: readability_results[x]["LIX"])
    df["SMOGIndex"] = df["text"].map(lambda x: readability_results[x]["SMOGIndex"])
    df["RIX"] = df["text"].map(lambda x: readability_results[x]["RIX"])
    df["DaleChallIndex"] = df["text"].map(lambda x: readability_results[x]["DaleChallIndex"])
    df["complex_words"] = df["text"].map(lambda x: readability_results[x]["complex_words"])
    df["complex_words_dc"] = df["text"].map(lambda x: readability_results[x]["complex_words_dc"])

    for i, lab in enumerate(class_mapping):
        df[lab] = df["text"].map(lambda x, i=i: topic_probs[idx(x)][i])
    df["topic_count"] = df["text"].map(lambda x: int(np.sum((topic_probs[idx(x)] >= 0.5) * 1)))
    df["topic_overall"] = df["text"].map(lambda x: class_mapping[int(np.argmax(topic_probs[idx(x)]))])

    for i, lab in enumerate(class_mapping_single):
        df["single_" + lab] = df["text"].map(lambda x, i=i: topic_single_probs[idx(x)][i])
    df["single_topic_count"] = df["text"].map(lambda x: int(np.sum((topic_single_probs[idx(x)] >= 0.5) * 1)))
    df["single_topic_overall"] = df["text"].map(
        lambda x: class_mapping_single[int(np.argmax(topic_single_probs[idx(x)]))]
    )

    df = m_mapping_category_values(df)

    gc.collect()
    return df