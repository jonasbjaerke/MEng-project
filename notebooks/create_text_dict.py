#!/usr/bin/env python
# coding: utf-8

# In[2]:


import json
import pandas as pd



# In[3]:


with open("../raw-data/userdata_mini.json", "r", encoding="utf-8") as f:
    raw_user_data = json.load(f)

with open("../raw-data/posts_mini.json", "r", encoding="utf-8") as f:
    posts = json.load(f)


# In[ ]:


# def build_post_text_dict(posts, raw_user_data, csv_path):


#     pairs_df = pd.read_csv(csv_path)

#     post_text_dict = {}


#     target_post_ids = set(pairs_df["P_id"].dropna().unique())

#     for uri in target_post_ids:
#         post = posts.get(uri)
#         if post:
#             text = post.get("text")
#             if text:
#                 post_text_dict[uri] = text


#     user_ids = set(pairs_df["A_id"].dropna().unique()) | \
#                set(pairs_df["S_id"].dropna().unique())

#     for user_id in user_ids:
#         user_data = raw_user_data.get(user_id)
#         if not user_data:
#             continue

#         for activity in user_data.get("history", []):
#             uri = activity.get("post_uri")
#             text = activity.get("text")

#             if uri and text:
#                 post_text_dict[uri] = text

#     return post_text_dict


# In[4]:


def build_post_text_dict(posts, raw_user_data):

    post_text_dict = {}

    # 1️⃣ Add all user history
    for user_data in raw_user_data.values():
        for activity in user_data.get("history", []):
            uri = activity.get("post_uri")
            text = activity.get("text")

            if uri and text:
                post_text_dict[uri] = text

    # 2️⃣ Add all original posts
    for uri, post in posts.items():
        text = post.get("text")
        if uri and text:
            post_text_dict[uri] = text

    return post_text_dict


# In[5]:


all_text_dict = build_post_text_dict(posts,raw_user_data)


# In[6]:


print(len(all_text_dict))


# In[8]:


with open("../raw-data/posts_text.jsonl", "w") as f:
    for uri, text in all_text_dict.items():
        f.write(json.dumps({
            "post_uri": uri,
            "text": text
        }) + "\n")

print("JSONL created.")

