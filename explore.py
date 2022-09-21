import pandas as pd
from utils import to_token_list, write_dataframe, print_examples, read_all_files
from constants import CONST


df = read_all_files(f"{CONST.data_root_path}/original", name="training_")

print_examples(df, 2)

df["tokenized"] = df["code"].apply(lambda x: to_token_list(x))

write_dataframe(
    df,
    f"{CONST.data_root_path}/processed",
    name="training_",
    bucket_step=500,
)
