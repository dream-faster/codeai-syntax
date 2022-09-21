#%%
import pandas as pd
from utils import to_token_list


df = pd.DataFrame([])
for i in range(99):
    df = pd.concat([df, pd.read_json(f"data/python_syntax/training_{i}.json")])

#%%
print(df["code"].iloc[0])
print(df["correct_code"].iloc[0])
print(df["wrong_code"].iloc[0])
# %%

df["tokenized"] = df["wrong_code"].apply(lambda x: to_token_list(x))
df
# %%
