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

    def reset(self):
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


class VariableTuner(StepTuner):
    def __init__(self) -> None:
        super().__init__()
        self.step_change = 0

    def reset(self):
        self.step_change = 0


class MSRTuner(VariableTuner):
    def __init__(
        self, decay_factor: float = 0.3, comparison_quantile: float = 0.25
    ) -> None:
        super().__init__()
        self.decay_factor = decay_factor
        self.comparison_quantile = comparison_quantile
        self.step_change = 0

    def adapt_step(
        self,
        mutation_step: float,
        population: np.ndarray,
        new_population: np.ndarray,
        evaluation_function: EvaluationFunction,
    ):
        evaluations = evaluation_function(population)
        baseline_evaluation = np.quantile(evaluations, self.comparison_quantile)

        new_evaluations = evaluation_function(new_population)
        successful_points = new_evaluations <= baseline_evaluation
        successes_count = np.count_nonzero(successful_points)
        population_size, dimension = population.shape
        success_measure = (
            2 / population_size * (successes_count - (population_size + 1) / 2)
        )

        self.step_change = (
            self.step_change * (1 - self.decay_factor)
            + success_measure * self.decay_factor
        )
        damping = 2 * (dimension - 1) / dimension
        return mutation_step * np.exp(self.step_change / damping)


class PSRTuner(VariableTuner):
    def __init__(
        self, decay_factor: float = 0.3, target_success_ratio: float = 0.25
    ) -> None:
        super().__init__()
        self.decay_factor = decay_factor
        self.target_success_ratio = target_success_ratio
        self.step_change = 0

    def adapt_step(
        self,
        mutation_step: float,
        population: np.ndarray,
        new_population: np.ndarray,
        evaluation_function: EvaluationFunction,
    ):
        evaluations = evaluation_function(population)
        evaluations = evaluations.reshape((-1, 1))
        new_evaluations = evaluation_function(new_population)
        new_evaluations = new_evaluations.reshape((-1, 1))

        # Add column marking which population evaluations originate from. 0 - current, 1 - new
        marked_evaluations = np.zeros((evaluations.shape[0], evaluations.shape[1] + 1))
        marked_evaluations[:, :-1] = evaluations
        marked_new_evaluations = np.ones(
            (new_evaluations.shape[0], new_evaluations.shape[1] + 1)
        )
        marked_new_evaluations[:, :-1] = new_evaluations
        merged_evaluations = np.concatenate(
            [marked_evaluations, marked_new_evaluations]
        )
        merged_evaluations = merged_evaluations[merged_evaluations[:, 0].argsort()]
        # Points with lower value of evaluation function (better) need to have higher rank for
        # calculation of success measure to work correctly.
        # When there are many successes (new points better than current ones), step should increase
        # to promote exploration. For this to happen, success measure needs to be high.
        all_ranks = np.arange(merged_evaluations.shape[0], 0, -1)

        ranks = all_ranks[merged_evaluations[:, 1] == 0]
        new_ranks = all_ranks[merged_evaluations[:, 1] == 1]

        population_size, dimension = population.shape
        success_measure = (
            new_ranks.sum() - ranks.sum()
        ) / population_size**2 - self.target_success_ratio
        self.step_change = (
            self.step_change * (1 - self.decay_factor)
            + success_measure * self.decay_factor
        )
        damping = 2 * (dimension - 1) / dimension
        return mutation_step * np.exp(self.step_change / damping)
