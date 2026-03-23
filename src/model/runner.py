

import pandas as pd
from .repost_predictor import RepostPredictor
from .xgboost import build_xgboost

df = pd.read_csv("data/processed/datasets/dataset.csv")

predictor = RepostPredictor(build_xgboost)


print("\n")
print("Mixed:")
print(predictor.evaluate_mixed(df))

print("\n")
print("In-Distribution:")
print(predictor.evaluate_in_distribution(df))

print("\n")
print("Out-of-Distribution:")
print(predictor.evaluate_out_of_distribution(df))

print("\n")
print("Feature importance:")
print(predictor.get_feature_gains())
