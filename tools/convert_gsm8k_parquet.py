import pandas as pd
import json

PARQUET_PATH = "/data/suyu/datasets/gsm8k/main/test-00000-of-00001.parquet"
OUT_PATH = "/home/suyu/projects/tree_search_framework/data/test.jsonl"

df = pd.read_parquet(PARQUET_PATH)

print("columns:", df.columns)

with open(OUT_PATH, "w") as f:
    for i, row in df.iterrows():
        item = {
            "question": row["question"],
            "answer": row["answer"]
        }
        f.write(json.dumps(item) + "\n")

print("saved to", OUT_PATH)
print("total samples:", len(df))