

import pandas as pd
from .repost_predictor import RepostPredictor
from .xgboost import build_xgboost

df = pd.read_csv("data/processed/datasets/dataset.csv")

predictor = RepostPredictor(build_xgboost)


print("Out-of-Distribution:")
print(predictor.evaluate_out_of_distribution(df))

print("Mixed:")
print(predictor.evaluate_mixed(df))

print("In-Distribution:")
print(predictor.evaluate_in_distribution(df))
