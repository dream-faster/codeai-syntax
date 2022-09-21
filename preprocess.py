import pandas as pd
from utils import to_token_list, write_dataframe, print_examples, read_all_files
from constants import CONST
from data.python_syntax.metadata import DataParams

df = read_all_files(f"{CONST.data_root_path}/original", name="training_")

print_examples(df, 2)

df[DataParams.token.value] = df[DataParams.code.value].apply(lambda x: to_token_list(x, key="type"))
df[DataParams.fix_location.value] = df[DataParams.metadata.value].apply(lambda x: x[DataParams.fix_location.value])

write_dataframe(
    df,
    f"{CONST.data_root_path}/processed",
    name="training_",
    bucket_step=500,
)
