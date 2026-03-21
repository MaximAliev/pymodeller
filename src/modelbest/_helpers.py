from collections import Counter
import itertools
from typing import Iterable, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from loguru import logger
from imblearn.datasets import make_imbalance
from sklearn.model_selection import train_test_split as tts
from typing import cast


def train_test_split(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if y is not None:
        try:
            return tuple(
                tts(X,
                    y,
                    random_state=42,
                    test_size=0.2,
                    stratify=y))
        except ValueError as exc:
            return tuple(
                tts(X,
                    y,
                    random_state=42,
                    test_size=0.2))   
    else:
        return tuple(
            tts(X,
                y,
                random_state=42,
                test_size=0.2))

def infer_positive_target_class(class_belongings: Counter) -> str:
    if len(class_belongings) > 2:
        raise ValueError("Multiclass problems currently not supported =(.")

    class_belongings_iterator = iter(sorted(cast(Iterable, class_belongings)))
    *_, pos_label = class_belongings_iterator
    logger.debug(f"Inferred positive class label: {pos_label}.")        
    
    return pos_label