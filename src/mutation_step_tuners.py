from abc import ABC, abstractmethod

import numpy as np

from .utility import EvaluationFunction


class StepTuner(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def adapt_step(
        self,
        mutation_step: float,
        population: np.ndarray = None,
        new_population: np.ndarray = None,
        evaluation_function: EvaluationFunction = None,
    ):
        pass


class ConstantStep(StepTuner):
    def adapt_step(
        self,
        mutation_step: float,
        population: np.ndarray = None,
        new_population: np.ndarray = None,
        evaluation_function: EvaluationFunction = None,
    ):
        return mutation_step
