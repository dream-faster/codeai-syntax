import pandas as pd
from utils import to_token_list, write_tokens, print_examples
from constants import CONST


df = pd.DataFrame([])
for i in range(5):
    df = pd.concat(
        [df, pd.read_json(f"{CONST.data_root_path}/original/training_{i}.json")]
    )

print_examples(df, 2)

df["tokenized"] = df["code"].apply(lambda x: to_token_list(x))

write_tokens(
    df,
    f"{CONST.data_root_path}/processed",
    name="training_",
    bucket_step=500,
)
