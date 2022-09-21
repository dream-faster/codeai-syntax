from enum import Enum
from models.pytorch_wrapped import PytorchModel
from models.linear import Linear
from type import PytorchConfig
from utils import read_all_files
from constants import CONST


from data.python_syntax.metadata import RawDataParams

# Load in data
df = read_all_files(f"{CONST.data_root_path}/processed", name="training_")

num_rows = (
    df[RawDataParams.correct_code.value].apply(lambda x: len(x.split["\n"])).to_list()
)
maximum_rows = max(num_rows)

config = PytorchConfig(hidden_size=64, output_size=maximum_rows, val_size=0.2)
new_model = PytorchModel("line-predictor", config, Linear)
