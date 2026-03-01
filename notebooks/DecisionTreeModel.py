#!/usr/bin/env python
# coding: utf-8

# In[259]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.model_selection import KFold


# In[263]:


def one_run(X, y, model):


    X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.3,
            stratify=y
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)


    return (
        f1_score(y_test, y_pred),
        confusion_matrix(y_test, y_pred)
    )


# In[280]:


def evaluate_model(X, y, model, n_runs=10):
    f1_scores = []
    conf_matrices = []

    for i in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.3,
            random_state=i,
            stratify=y
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # F1
        f1_scores.append(f1_score(y_test, y_pred))

        # Confusion matrix
        conf_matrices.append(confusion_matrix(y_test, y_pred))

    #print(f"Size of training set: {len(y_train)}", f"Size of testing set: {len(y_test)}")
    # Convert to numpy array for averaging
    conf_matrices = np.array(conf_matrices)

    return (
        np.mean(f1_scores),
        np.std(f1_scores),
        np.mean(conf_matrices, axis=0),
        np.std(conf_matrices, axis=0)
    )


# In[ ]:


def test_many_datasets(s_features,id_cols):
    model = XGBClassifier(
    max_depth=8,
    learning_rate=0.1,
    n_estimators=100,
    scale_pos_weight=3,
    eval_metric="logloss",
    objective="binary:logistic",
    tree_method="hist"
    )

    f1s = []
    matrices = []
    for i in range(1,10):
        dataset = pd.read_csv(f"mini_1:5_#{i}.csv")
        X = dataset.drop(columns=id_cols + ["label"] + s_features)
        y = dataset["label"]
        X = X.fillna(0)

        score, conf_matrix = one_run(X,y,model)
        f1s.append(score)
        matrices.append(conf_matrix)

    mean_f1= np.mean(f1s)
    std_f1 = np.std(f1s)
    avg_conf = np.mean(matrices,axis=0)
    std_conf = np.std(matrices,axis=0)
    print("ONLY R FEATURES + remove U-HA_R_RetweetPercent and  U-HA_R_AverageInterval")
    print("-" * 40)

    print("Mixed 1:5 for 9 different datasets (same pos but random neg):")
    print("-" * 40)

    print(f"F1: {mean_f1:.4f} ± {std_f1:.4f}")

    print("Confusion Matrix")
    print(f"TN: {avg_conf[0,0]:.1f} ± {std_conf[0,0]:.1f}")
    print(f"FP: {avg_conf[0,1]:.1f} ± {std_conf[0,1]:.1f}")
    print(f"FN: {avg_conf[1,0]:.1f} ± {std_conf[1,0]:.1f}")
    print(f"TP: {avg_conf[1,1]:.1f} ± {std_conf[1,1]:.1f}")
    print("-" * 40)


    f1s = []
    matrices = []
    for i in range(1,10):
        dataset = pd.read_csv(f"mini_1:1_#{i}.csv")
        X = dataset.drop(columns=id_cols + ["label"] + s_features)
        y = dataset["label"]
        X = X.fillna(0)

        score, conf_matrix = one_run(X,y,model)
        f1s.append(score)
        matrices.append(conf_matrix)

    mean_f1= np.mean(f1s)
    std_f1 = np.std(f1s)
    avg_conf = np.mean(matrices,axis=0)
    std_conf = np.std(matrices,axis=0)

    print("Mixed 1:1 for 9 different datasets (same pos but random neg):")
    print("-" * 40)

    print(f"F1: {mean_f1:.4f} ± {std_f1:.4f}")

    print("Confusion Matrix")
    print(f"TN: {avg_conf[0,0]:.1f} ± {std_conf[0,0]:.1f}")
    print(f"FP: {avg_conf[0,1]:.1f} ± {std_conf[0,1]:.1f}")
    print(f"FN: {avg_conf[1,0]:.1f} ± {std_conf[1,0]:.1f}")
    print(f"TP: {avg_conf[1,1]:.1f} ± {std_conf[1,1]:.1f}")
    print("-" * 40)


# In[ ]:


def in_distribution_hashtags(df, n_runs=10):
    print("In-Distribution")
    results = {}
    


    model = XGBClassifier(
    max_depth=8,
    learning_rate=0.1,
    n_estimators=100,
    scale_pos_weight=3,
    reg_lambda=2,
    eval_metric="logloss",
    objective="binary:logistic",
    tree_method="hist"
    )
    
    for tag in df["hashtag"].unique():

        df_tag = df[df["hashtag"] == tag]


        X = df_tag.drop(columns=["label", "hashtag", "A_id", "S_id", "P_id"])
        y = df_tag["label"]

        mean_f1, std_f1, avg_conf = evaluate_model(X, y, model, n_runs=n_runs)

        results[tag] = (mean_f1, std_f1)

        print(f"Hashtag: {tag}")
        print(f"F1: {mean_f1:.4f} ± {std_f1:.4f}")
        print("-" * 40)



# In[ ]:


def out_of_distribution_hashtags(df):

    hashtags = df["hashtag"].unique()
    print("Out-of-Distribution")
    results = {}
    

    model = XGBClassifier(
    max_depth=8,
    learning_rate=0.1,
    n_estimators=100,
    scale_pos_weight=3,
    reg_lambda=2,
    eval_metric="logloss",
    objective="binary:logistic",
    tree_method="hist"
    )
        
    for tag in hashtags:

        test_df = df[df["hashtag"] == tag].copy()
        train_df = df[df["hashtag"] != tag].copy()
        print(f"Size of training set: {len(train_df)}", f"Size of testing set: {len(test_df)}")

        X_test = test_df.drop(columns=["label", "hashtag", "A_id", "S_id", "P_id"])
        y_test = test_df["label"]

        f1_scores = []

        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        for train_index, _ in kf.split(train_df):

            train_subset = train_df.iloc[train_index]

            X_train = train_subset.drop(columns=["label", "hashtag", "A_id", "S_id", "P_id"])
            y_train = train_subset["label"]

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            f1_scores.append(f1_score(y_test, y_pred))

        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)

        print(f"Hashtag: {tag}")
        print(f"F1: {mean_f1:.4f} ± {std_f1:.4f}")
        print("-" * 40)

        results[tag] = (mean_f1, std_f1)





# In[278]:


df_1_1 = pd.read_csv("mini_1:1_#1.csv")
df_1_5 = pd.read_csv("mini_1:5_#1.csv")


# In[147]:


out_of_distribution_hashtags(df_1_1)


# In[146]:


out_of_distribution_hashtags(df_1_5)


# In[148]:


in_distribution_hashtags(df_1_1)


# In[145]:


in_distribution_hashtags(df_1_5)


# In[ ]:


s_features = [
    "U-HA_S_MentionR",
    "U-HA_S_MentionPerR",
    "U-P_S_AccountAge",
    "U-P_S_FollowerNum",
    "U-P_S_FolloweeNum",
    "U-P_S_TweetNum",
    "U-P_S_SpreadActivity",
    "U-P_S_FollowerNumDay",
    "U-P_S_FolloweeNumDay",
    "U-P_S_TweetNumDay",
    "U-P_S_ProfileUrl",
    "U-HA_S_RetweetPercent",
    "U-HA_S_AverageInterval",
    "U-HA_S_RetweetedRate",
    "U-HA_S_QuotedRate",
    "U-HA_S_RepliedRate",
    "U-HA_S_LikedRate",
    "U-HA_S_TweetNum"
]

r_features = [
    "U-P_R_FollowS",
    "U-HA_R_MentionS",
    "U-HA_R_MentionPerS",
    "U-HA-R_repostsS",
    "U-P_R_activeBeforeP",
    "U-P_R_AccountAge",
    "U-P_R_FollowerNum",
    "U-P_R_FolloweeNum",
    "U-P_R_TweetNum",
    "U-P_R_SpreadActivity",
    "U-P_R_FollowerNumDay",
    "U-P_R_FolloweeNumDay",
    "U-P_R_TweetNumDay",
    "U-P_R_ProfileUrl",
    "U-HA_R_RetweetPercent",
    "U-HA_R_AverageInterval",
    "U-HA_R_RetweetedRate",
    "U-HA_R_QuotedRate",
    "U-HA_R_RepliedRate",
    "U-HA_R_LikedRate",
    "U-HA_R_TweetNum"
]

id_cols = ["A_id", "S_id", "P_id", "hashtag"]


model = XGBClassifier(
    max_depth=8,
    learning_rate=0.1,
    n_estimators=100,
    scale_pos_weight=3,
    eval_metric="logloss",
    objective="binary:logistic",
    tree_method="hist"
    )


print("ONLY R FEATURES")
X = df_1_5.drop(columns=id_cols + ["label"] + s_features)
y = df_1_5["label"]
X = X.fillna(0)

mean_f1, std_f1, avg_conf, std_conf = evaluate_model(X, y, model, n_runs=10)

print("Mixed 1:5:")
print("-" * 40)

print(f"F1: {mean_f1:.4f} ± {std_f1:.4f}")

print("Confusion Matrix")
print(f"TN: {avg_conf[0,0]:.1f} ± {std_conf[0,0]:.1f}")
print(f"FP: {avg_conf[0,1]:.1f} ± {std_conf[0,1]:.1f}")
print(f"FN: {avg_conf[1,0]:.1f} ± {std_conf[1,0]:.1f}")
print(f"TP: {avg_conf[1,1]:.1f} ± {std_conf[1,1]:.1f}")
print("-" * 40)



X = df_1_1.drop(columns=id_cols + ["label"] + s_features)
y = df_1_1["label"]
X = X.fillna(0)


mean_f1, std_f1, avg_conf, std_conf = evaluate_model(X, y, model, n_runs=10)

print("Mixed 1:1:")
print("-" * 40)

print(f"F1: {mean_f1:.4f} ± {std_f1:.4f}")
print("Confusion Matrix")
print(f"TN: {avg_conf[0,0]:.1f} ± {std_conf[0,0]:.1f}")
print(f"FP: {avg_conf[0,1]:.1f} ± {std_conf[0,1]:.1f}")
print(f"FN: {avg_conf[1,0]:.1f} ± {std_conf[1,0]:.1f}")
print(f"TP: {avg_conf[1,1]:.1f} ± {std_conf[1,1]:.1f}")
print("-" * 40)


# In[276]:


booster = model.get_booster()

# Get raw gain importance
importance_dict = booster.get_score(importance_type="gain")

# Convert to DataFrame
importance_df = pd.DataFrame(
    list(importance_dict.items()),
    columns=["feature", "gain"]
)

# Sort by gain
importance_df = importance_df.sort_values(
    by="gain",
    ascending=False
).reset_index(drop=True)

# Convert gain to percentage importance
total_gain = importance_df["gain"].sum()
importance_df["gain"] = (
    importance_df["gain"] / total_gain
)

print(importance_df)

