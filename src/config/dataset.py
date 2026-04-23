from dataclasses import dataclass


@dataclass
class DatasetConfig:
    
    builder: str = "Hybrid"   # "M", "U", "Hybrid", "Bert"
    neg_per_pos: int = 1

    posts_filename: str = "postsFinal.json"
    users_filename: str = "usersFinal.json"
    output_filename: str = "dataset.csv"

    dataset_seed: int = 42


HYBRID_1TO1 = DatasetConfig(
    builder="Hybrid",
    neg_per_pos=1,
    output_filename="dataset_hybrid_1to1.csv",
)

HYBRID_1TO5 = DatasetConfig(
    builder="Hybrid",
    neg_per_pos=5,
    output_filename="dataset_hybrid_1to5.csv",
)

USER_1TO1 = DatasetConfig(
    builder="U",
    neg_per_pos=1,
    output_filename="dataset_user_1to1.csv",
)

USER_1TO5 = DatasetConfig(
    builder="U",
    neg_per_pos=5,
    output_filename="dataset_user_1to5.csv",
)

MESSAGE_1TO1 = DatasetConfig(
    builder="M",
    neg_per_pos=1,
    output_filename="dataset_message_1to1.csv",
    dataset_seed= 878
)

MESSAGE_1TO5 = DatasetConfig(
    builder="M",
    neg_per_pos=5,
    output_filename="dataset_message_1to5.csv",

)

BERT_1TO1 = DatasetConfig(
    builder="Bert",
    neg_per_pos=1,
    output_filename="dataset_bert_1to1.csv",
)

BERT_1TO5 = DatasetConfig(
    builder="Bert",
    neg_per_pos=5,
    output_filename="dataset_bert_1to5.csv",
)


DATASET_CONFIGS = {
    "HYBRID_1TO1": HYBRID_1TO1,
    "HYBRID_1TO5": HYBRID_1TO5,
    "USER_1TO1": USER_1TO1,
    "USER_1TO5": USER_1TO5,
    "MESSAGE_1TO1": MESSAGE_1TO1,
    "MESSAGE_1TO5": MESSAGE_1TO5,
    "BERT_1TO1": BERT_1TO1,
    "BERT_1TO5": BERT_1TO5,
}