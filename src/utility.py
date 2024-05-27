from typing import Callable

import numpy as np

EvaluationFunction = Callable[
    [np.ndarray[np.ndarray[np.floating]]], np.ndarray[np.floating]
]
