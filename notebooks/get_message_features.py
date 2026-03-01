#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from tqdm import tqdm
import os

from m_features import add_all_m_features


# In[2]:


import gc
import torch


# In[3]:


INPUT_FILE = "../raw-data/posts-text_dict.jsonl"     
OUTPUT_DIR = "../message_features"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CHUNK_SIZE = 5000  # safer for large transformer models
BATCH_SIZE = 128      # increase if GPU


# In[4]:


total_lines = sum(1 for _ in open(INPUT_FILE))
total_chunks = (total_lines // CHUNK_SIZE) + 1

print(f"Total posts: {total_lines}")
print(f"Total chunks: {total_chunks}")


# In[ ]:


reader = pd.read_json(INPUT_FILE, lines=True, chunksize=CHUNK_SIZE)

for i, chunk in tqdm(enumerate(reader), total=total_chunks, desc="Chunks"):

    # Ensure required columns exist
    if "text" not in chunk.columns:
        raise ValueError("JSONL must contain a 'text' column")

    # Apply full feature pipeline
    df_features = add_all_m_features(
        chunk,
        batch_size=BATCH_SIZE
    )

    # Save chunk
    output_path = os.path.join(OUTPUT_DIR, f"features_chunk_{i}.parquet")
    df_features.to_parquet(output_path, index=False)

    del df_features
    gc.collect()
    torch.mps.empty_cache()


print("\nAll chunks processed successfully.")


# In[11]:


import glob
import pandas as pd

files = sorted(glob.glob("../message_features/features_chunk_*.parquet"))

for f in files[:1]:
    df_chunk = pd.read_parquet(f)

    # Do analysis here
    print(df_chunk)
    del df_chunk

