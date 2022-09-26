from enum import Enum
from typing import Callable, List

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def __wrap_sklearn_scorer(scorer: Callable, *args, **kwargs) -> Callable:
    def wrapper(y_true, predicted_probs) -> float:
        return scorer(y_true, [item[0] for item in predicted_probs], *args, **kwargs)

    return wrapper


class CTypes(Enum):
    micro = "micro"
    macro = "macro"
    weighted = "weighted"


multiclass_classification_metrics = [
    (
        "f1_micro",
        __wrap_sklearn_scorer(f1_score, average=CTypes.micro.value),
    ),
    (
        "f1_macro",
        __wrap_sklearn_scorer(f1_score, average=CTypes.macro.value),
    ),
    (
        "f1_weighted",
        __wrap_sklearn_scorer(f1_score, average=CTypes.weighted.value),
    ),
    (
        "accuracy",
        __wrap_sklearn_scorer(accuracy_score),
    ),
    (
        "precision_micro",
        __wrap_sklearn_scorer(precision_score, average=CTypes.micro.value),
    ),
    (
        "precision_macro",
        __wrap_sklearn_scorer(precision_score, average=CTypes.macro.value),
    ),
    (
        "precision_weighted",
        __wrap_sklearn_scorer(precision_score, average=CTypes.weighted.value),
    ),
    (
        "recall_micro",
        __wrap_sklearn_scorer(recall_score, average=CTypes.micro.value),
    ),
    (
        "recall_macro",
        __wrap_sklearn_scorer(recall_score, average=CTypes.macro.value),
    ),
    (
        "recall_weighted",
        __wrap_sklearn_scorer(recall_score, average=CTypes.weighted.value),
    ),
    ("report", __wrap_sklearn_scorer(classification_report)),
]


binary_classification_metrics = [
    ("f1_binary", __wrap_sklearn_scorer(f1_score)),
    ("f1_binary_class_0", __wrap_sklearn_scorer(f1_score, pos_label=0)),
    ("precision_binary", __wrap_sklearn_scorer(precision_score)),
    ("precision_binary_class_0", __wrap_sklearn_scorer(precision_score, pos_label=0)),
    ("recall_binary", __wrap_sklearn_scorer(recall_score)),
    ("recall_binary_class_0", __wrap_sklearn_scorer(recall_score, pos_label=0)),
    ("roc_auc", __wrap_sklearn_scorer(roc_auc_score)),
]

classification_metrics = (
    binary_classification_metrics + multiclass_classification_metrics
)
