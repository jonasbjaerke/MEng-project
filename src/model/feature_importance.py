import os
import pandas as pd
import matplotlib.pyplot as plt

from .repost_predictor import RepostPredictor
from .xgboost import build_xgboost


def extract_mean_f1(result):
    if "f1_mean" in result:
        return float(result["f1_mean"])

    f1s = [float(v["f1_mean"]) for v in result.values()]
    return sum(f1s) / len(f1s)


def get_next_removable_feature(gain_df, ignored_features):
    """
    Return the highest-gain feature that:
    - is not already ignored
    - does not start with 'M'
    """
    for _, row in gain_df.iterrows():
        feature = str(row["feature"])
        if feature in ignored_features:
            continue
        if feature.startswith("M"):
            continue
        return feature
    return None


def main():
    df = pd.read_csv("data/processed/datasets/U1.csv")

    results = []
    ignored_features = []
    n_runs = 50

    os.makedirs("data/plots", exist_ok=True)

    for run_idx in range(n_runs):
        predictor = RepostPredictor(build_xgboost)

        if ignored_features:
            predictor.ignore_features(ignored_features)

        print(f"\n=== Run {run_idx + 1} / {n_runs} ===")
        print(f"Ignored features: {ignored_features}")

        mixed_result = predictor.evaluate_mixed(df)
        ood_result = predictor.evaluate_out_of_distribution(df)

        mixed_f1 = extract_mean_f1(mixed_result)
        ood_avg_f1 = extract_mean_f1(ood_result)

        gain_df = predictor.get_feature_gains()
        top_feature = get_next_removable_feature(gain_df, ignored_features)

        results.append({
            "run": run_idx + 1,
            "n_removed": len(ignored_features),
            "removed_so_far": ", ".join(ignored_features) if ignored_features else "None",
            "mixed_f1": mixed_f1,
            "ood_avg_f1": ood_avg_f1,
            "top_feature_removed_next": top_feature,
        })

        print(f"Mixed F1: {mixed_f1:.4f}")
        print(f"OOD Avg F1: {ood_avg_f1:.4f}")
        print(f"Top removable feature next: {top_feature}")

        if top_feature is None:
            break

        ignored_features.append(top_feature)

    results_df = pd.DataFrame(results)

    plt.figure(figsize=(10, 5))
    plt.plot(results_df["n_removed"], results_df["mixed_f1"], marker="o", label="Mixed F1")
    plt.plot(results_df["n_removed"], results_df["ood_avg_f1"], marker="o", label="OOD Avg F1")

    plt.xlabel("Number of Top Features Removed")
    plt.ylabel("F1 Score")
    plt.title("Effect of Removing Top Features")

    x_vals = results_df["n_removed"].tolist()
    if len(x_vals) > 10:
        step = max(1, len(x_vals) // 10)
        xticks = x_vals[::step]
        if x_vals[-1] not in xticks:
            xticks.append(x_vals[-1])
        plt.xticks(xticks)
    else:
        plt.xticks(x_vals)

    plt.legend()
    plt.tight_layout()
    plt.savefig("data/plots/feature_removal_f1.png", dpi=300)
    plt.close()

    print("\nSaved:")
    print("- plots/feature_removal_f1.png")
    print("\nResults:")
    print(results_df)


if __name__ == "__main__":
    main()