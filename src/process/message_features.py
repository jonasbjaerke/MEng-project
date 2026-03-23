import os
import gc
import re
import time

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
    text = re.sub(r"(@[A-Za-z0-9_]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text)

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
    return text.lower().strip()


def grammar_p(text):
    text = str(text)
    text = re.sub(r"RT @\w+: ", " ", text)
    text = re.sub(r"(@[A-Za-z0-9_]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text)

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
    try:
        sentences = nltk.sent_tokenize(text)
        return "\n".join(sentences)
    except Exception:
        return text


def safe_readability(tokenized_text):
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


def get_grammar_score(text):
    tool = get_grammar_tool()
    if tool is None:
        return [np.nan, np.nan]

    try:
        sentences = nltk.tokenize.sent_tokenize(text)
    except Exception:
        return [np.nan, np.nan]

    if len(sentences) == 0:
        return [np.nan, np.nan]

    scores_word_based_sentence = []
    scores_sentence_based_sentence = []

    for sentence in sentences:
        try:
            matches = tool.check(sentence)
            count_errors = len(matches)
        except Exception:
            return [np.nan, np.nan]

        scores_sentence_based_sentence.append(min(count_errors, 1))
        scores_word_based_sentence.append(count_errors)

    try:
        word_count = len(nltk.tokenize.word_tokenize(text))
    except Exception:
        word_count = 0

    score_word_based = np.nan if word_count == 0 else 1 - (np.sum(scores_word_based_sentence) / word_count)
    sentence_count = len(sentences)
    score_sentence_based = 1 - (np.sum(scores_sentence_based_sentence) / sentence_count)
    return [score_word_based, score_sentence_based]


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


def add_all_m_features(df, text_col="full_text", batch_size=128):
    start_time = time.perf_counter()

    df = df.copy()
    df = df.dropna(subset=[text_col]).copy()
    df[text_col] = df[text_col].astype(str)

    unique_text = df[text_col].unique()

    text_len, word_count = {}, {}
    sentiment_scores = {}
    grammar_score = {}
    subjectivity_score, polarity_score = {}, {}
    emo_results, hs_results = {}, {}
    irony_scores, offensive_scores = {}, {}
    emoji_count = {}
    readability_results = {}
    topic_scores, topic_pred = {}, {}
    topic_scores_single, topic_pred_single = {}, {}

    e_count = 0
    remove_list = []

    def log_processing_error(step, idx, text, e):
        print(f"[ERROR {step}] text #{idx}: {repr(str(text)[:200])}")
        print(f"  error: {type(e).__name__}: {e}")

    for idx, text in enumerate(unique_text, 1):
        text = str(text)

        try:
            grammar_text = grammar_p(text)
            berkem_text = berkem_preprocess(text)
        except Exception as e:
            log_processing_error("preprocess", idx, text, e)
            e_count += 1
            remove_list.append(text)
            continue

        try:
            text_len[text] = len(berkem_text)
            word_count[text] = len(str(berkem_text).split())
        except Exception as e:
            log_processing_error("length/word_count", idx, text, e)
            e_count += 1
            remove_list.append(text)
            continue

        try:
            sentiment_scores[text] = sia.polarity_scores(berkem_text)
        except Exception as e:
            log_processing_error("sentiment", idx, text, e)
            e_count += 1
            remove_list.append(text)
            continue

        try:
            grammar_score[text] = get_grammar_score(grammar_text)
        except Exception as e:
            log_processing_error("grammar", idx, text, e)
            e_count += 1
            remove_list.append(text)
            continue

        try:
            subjectivity_score[text] = TextBlob(text).sentiment[1]
            polarity_score[text] = TextBlob(text).sentiment[0]
        except Exception as e:
            log_processing_error("textblob", idx, text, e)
            e_count += 1
            remove_list.append(text)
            continue

        try:
            emo_results[text] = emotion_analyzer.predict(preprocess_tweet(text))
        except Exception as e:
            log_processing_error("emotion", idx, text, e)
            e_count += 1
            remove_list.append(text)
            continue

        try:
            hs_results[text] = hate_speech_analyzer.predict(preprocess_tweet(text))
        except Exception as e:
            log_processing_error("hate_speech", idx, text, e)
            e_count += 1
            remove_list.append(text)
            continue

        try:
            encoded_input_irony = tokenizer_irony(
                roberta_preprocess(text),
                return_tensors="pt",
                truncation=True,
                max_length=128,
            )
            encoded_input_irony = {k: v.to(DEVICE) for k, v in encoded_input_irony.items()}
            output_irony = model_irony(**encoded_input_irony)
            irony_scores[text] = softmax(output_irony.logits[0].detach().cpu().numpy())
        except Exception as e:
            log_processing_error("irony", idx, text, e)
            e_count += 1
            remove_list.append(text)
            continue

        try:
            encoded_input_offensive = tokenizer_offensive(
                roberta_preprocess(text),
                return_tensors="pt",
                truncation=True,
                max_length=128,
            )
            encoded_input_offensive = {k: v.to(DEVICE) for k, v in encoded_input_offensive.items()}
            output_offensive = model_offensive(**encoded_input_offensive)
            offensive_scores[text] = softmax(output_offensive.logits[0].detach().cpu().numpy())
        except Exception as e:
            log_processing_error("offensive", idx, text, e)
            e_count += 1
            remove_list.append(text)
            continue

        try:
            encoded_input_emoji = tokenizer_emoji(
                roberta_preprocess(text),
                return_tensors="pt",
                truncation=True,
                max_length=128,
            )
            encoded_input_emoji = {k: v.to(DEVICE) for k, v in encoded_input_emoji.items()}
            output_emoji = model_emoji(**encoded_input_emoji)
            emoji_scores_local = softmax(output_emoji.logits[0].detach().cpu().numpy())
            ranking = np.argsort(emoji_scores_local)[::-1]
            emoji_count[text] = int(ranking[0])
        except Exception as e:
            log_processing_error("emoji", idx, text, e)
            e_count += 1
            remove_list.append(text)
            continue

        try:
            readability_results[text] = safe_readability(read_tokenized(berkem_text))
        except Exception as e:
            log_processing_error("readability", idx, text, e)
            e_count += 1
            readability_results[text] = None

        try:
            tokens_topic = tokenizer_topic(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
            )
            tokens_topic = {k: v.to(DEVICE) for k, v in tokens_topic.items()}
            output_topic = model_topic(**tokens_topic)
            topic_scores[text] = expit(output_topic.logits[0].detach().cpu().numpy())
            topic_pred[text] = (topic_scores[text] >= 0.5) * 1
        except Exception as e:
            log_processing_error("topic-multi", idx, text, e)
            e_count += 1
            topic_scores[text] = None
            topic_pred[text] = None
            remove_list.append(text)
            continue

        try:
            tokens_topic_single = tokenizer_topic_single(
                roberta_preprocess(text),
                return_tensors="pt",
                truncation=True,
                max_length=128,
            )
            tokens_topic_single = {k: v.to(DEVICE) for k, v in tokens_topic_single.items()}
            output_topic_single = model_topic_single(**tokens_topic_single)
            topic_scores_single[text] = expit(output_topic_single.logits[0].detach().cpu().numpy())
            topic_pred_single[text] = (topic_scores_single[text] >= 0.5) * 1
        except Exception as e:
            log_processing_error("topic-single", idx, text, e)
            e_count += 1
            topic_scores_single[text] = None
            topic_pred_single[text] = None
            remove_list.append(text)
            continue

    df = df[~df[text_col].astype(str).isin(remove_list)].copy()

    df["text_len"] = df[text_col].apply(lambda x: text_len[str(x)])
    df["word_count"] = df[text_col].apply(lambda x: word_count[str(x)])

    df["neg"] = df[text_col].apply(lambda x: sentiment_scores[str(x)]["neg"])
    df["neu"] = df[text_col].apply(lambda x: sentiment_scores[str(x)]["neu"])
    df["pos"] = df[text_col].apply(lambda x: sentiment_scores[str(x)]["pos"])
    df["compound"] = df[text_col].apply(lambda x: sentiment_scores[str(x)]["compound"])
    df["sentiment_overall"] = df[text_col].apply(
        lambda x: assign_sentiment(
            sentiment_scores[str(x)]["neg"],
            sentiment_scores[str(x)]["pos"]
        )
    )

    df["grammar-word-score"] = df[text_col].apply(lambda x: grammar_score[str(x)][0])
    df["grammar-sentence-score"] = df[text_col].apply(lambda x: grammar_score[str(x)][1])

    df["subjectivity"] = df[text_col].apply(lambda x: subjectivity_score[str(x)])
    df["polarity"] = df[text_col].apply(lambda x: polarity_score[str(x)])

    df["emo_overall"] = df[text_col].apply(lambda x: emo_results[str(x)].output)
    df["emo_anger"] = df[text_col].apply(lambda x: emo_results[str(x)].probas["anger"])
    df["emo_joy"] = df[text_col].apply(lambda x: emo_results[str(x)].probas["joy"])
    df["emo_fear"] = df[text_col].apply(lambda x: emo_results[str(x)].probas["fear"])
    df["emo_disgust"] = df[text_col].apply(lambda x: emo_results[str(x)].probas["disgust"])
    df["emo_surprise"] = df[text_col].apply(lambda x: emo_results[str(x)].probas["surprise"])
    df["emo_sadness"] = df[text_col].apply(lambda x: emo_results[str(x)].probas["sadness"])
    df["emo_others"] = df[text_col].apply(lambda x: emo_results[str(x)].probas["others"])

    df["hs_aggressive"] = df[text_col].apply(lambda x: hs_results[str(x)].probas["aggressive"])
    df["hs_hateful"] = df[text_col].apply(lambda x: hs_results[str(x)].probas["hateful"])
    df["hs_targeted"] = df[text_col].apply(lambda x: hs_results[str(x)].probas["targeted"])
    df["hs_count"] = df[text_col].apply(lambda x: len(hs_results[str(x)].output))

    df["irony"] = df[text_col].apply(lambda x: 0 if irony_scores[str(x)][1] < 0.5 else 1)
    df["offensive"] = df[text_col].apply(lambda x: offensive_scores[str(x)][1])

    df["emoji"] = df[text_col].apply(lambda x: emoji_count[str(x)])

    df["Kincaid"] = df[text_col].apply(
        lambda x: np.nan if readability_results[str(x)] is None
        else readability_results[str(x)]["readability grades"]["Kincaid"]
    )
    df["ARI"] = df[text_col].apply(
        lambda x: np.nan if readability_results[str(x)] is None
        else readability_results[str(x)]["readability grades"]["ARI"]
    )
    df["Coleman-Liau"] = df[text_col].apply(
        lambda x: np.nan if readability_results[str(x)] is None
        else readability_results[str(x)]["readability grades"]["Coleman-Liau"]
    )
    df["FleschReadingEase"] = df[text_col].apply(
        lambda x: np.nan if readability_results[str(x)] is None
        else readability_results[str(x)]["readability grades"]["FleschReadingEase"]
    )
    df["GunningFogIndex"] = df[text_col].apply(
        lambda x: np.nan if readability_results[str(x)] is None
        else readability_results[str(x)]["readability grades"]["GunningFogIndex"]
    )
    df["LIX"] = df[text_col].apply(
        lambda x: np.nan if readability_results[str(x)] is None
        else readability_results[str(x)]["readability grades"]["LIX"]
    )
    df["SMOGIndex"] = df[text_col].apply(
        lambda x: np.nan if readability_results[str(x)] is None
        else readability_results[str(x)]["readability grades"]["SMOGIndex"]
    )
    df["RIX"] = df[text_col].apply(
        lambda x: np.nan if readability_results[str(x)] is None
        else readability_results[str(x)]["readability grades"]["RIX"]
    )
    df["DaleChallIndex"] = df[text_col].apply(
        lambda x: np.nan if readability_results[str(x)] is None
        else readability_results[str(x)]["readability grades"]["DaleChallIndex"]
    )
    df["complex_words"] = df[text_col].apply(
        lambda x: np.nan if readability_results[str(x)] is None
        else readability_results[str(x)]["sentence info"]["complex_words"]
    )
    df["complex_words_dc"] = df[text_col].apply(
        lambda x: np.nan if readability_results[str(x)] is None
        else readability_results[str(x)]["sentence info"]["complex_words_dc"]
    )

    for i in range(len(class_mapping)):
        df[class_mapping[i]] = df[text_col].apply(lambda x, i=i: topic_scores[str(x)][i])

    df["topic_count"] = df[text_col].apply(lambda x: int(sum(topic_pred[str(x)])))
    df["topic_overall"] = df[text_col].apply(
        lambda x: class_mapping[
            topic_scores[str(x)].tolist().index(max(topic_scores[str(x)]))
        ]
    )

    for i in range(len(class_mapping_single)):
        df["single_" + class_mapping_single[i]] = df[text_col].apply(
            lambda x, i=i: topic_scores_single[str(x)][i]
        )

    df["single_topic_count"] = df[text_col].apply(lambda x: int(sum(topic_pred_single[str(x)])))
    df["single_topic_overall"] = df[text_col].apply(
        lambda x: class_mapping_single[
            topic_scores_single[str(x)].tolist().index(max(topic_scores_single[str(x)]))
        ]
    )

    df = m_mapping_category_values(df)

    gc.collect()

    finish_time = time.perf_counter()
    print(f"processed {len(df)} rows")
    print(f"removed {len(set(remove_list))} unique texts because of processing errors")
    print(f"total processing errors: {e_count}")
    print(f"finished in {(finish_time - start_time) / 60} minutes")

    return df