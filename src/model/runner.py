import argparse
import pandas as pd

from .DT_repost_predictor import RepostPredictor
from .bert_repost_predictor import BertRepostPredictor, BertConfig
from .xgboost import build_xgboost
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=["dt", "bert"],
        default="dt",
        help="Which model family to run: dt or bert",
    )
    parser.add_argument(
        "--file",
        type=str,
        default="dataset.csv",
        help="Name of dataset file (located in data/processed/datasets/)"
    )
    parser.add_argument(
        "--eval",
        type=str,
        choices=["mixed", "ood", "id", "all"],
        default="all",
        help="Which evaluation to run",
    )

    #bert config
    parser.add_argument("--bert_model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)

    args = parser.parse_args()

    BASE_DIR = "data/processed/datasets"
    data_path = os.path.join(BASE_DIR, args.file)

    df = pd.read_csv(data_path)

    if args.model == "dt":
        predictor = RepostPredictor(build_xgboost)
    elif args.model == "bert":
        predictor = BertRepostPredictor(
            BertConfig(
                model_name=args.bert_model_name,
                max_length=args.max_length,
                batch_size=args.batch_size,
                num_train_epochs=args.epochs,
            )
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    print(f"\nRunning model: {args.model}")
    print(f"\nRunning file: {args.file}")

    if args.eval in ["mixed", "all"]:
        print("\nMixed:")
        print(predictor.evaluate_mixed(df))

    if args.eval in ["ood", "all"]:
        print("\nOut-of-Distribution:")
        print(predictor.evaluate_out_of_distribution(df))

    if args.eval in ["id", "all"]:
        print("\nIn-Distribution:")
        print(predictor.evaluate_in_distribution(df))

    if args.model == "dt":
        print("\nFeature importance:")
        gains_df = predictor.get_feature_gains()
        print(gains_df)

        output_dir = "results/DT/Feature_analysis"
        os.makedirs(output_dir, exist_ok=True)

        filename = f"{os.path.splitext(args.file)[0]}_feature_gains.csv"
        output_path = os.path.join(output_dir, filename)

        gains_df.to_csv(output_path, index=False)

        print(f"\nFeature gains saved to: {output_path}")


if __name__ == "__main__":
    main()