from preprocess import preprocess
from run_train import train_test
from run_inference import run_inference
from type import dev_config

# 1. First we need to run the preprocessing on the dataset
preprocess()


def test_train_testing():
    train_test(dev_config)


def test_inference():
    preds, probs = run_inference()

    assert len(preds) > 0, "No prediction where thrown back."
    assert len(probs) > 0, "Probabilities were not included in the prediction."
