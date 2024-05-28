import numpy as np

from .mutation_step_tuners import ConstantStep, StepTuner
from .utility import EvaluationFunction


class DifferentialEvolution:
    def __init__(
        self,
        evaluation_function: EvaluationFunction,
        initial_mutation_step: float = 0.5,
        crossover_rate: float = 0.9,
        mutation_step_tuner: StepTuner = ConstantStep(),
    ) -> None:
        self.evaluation_function = evaluation_function
        self.crossover_rate = crossover_rate
        self.initial_mutation_step = initial_mutation_step
        self.mutation_step_tuner = mutation_step_tuner

    def evolve(self, initial_population: np.ndarray, number_of_generations: int = 20):
        population = initial_population.copy()
        self.mutation_step_tuner.reset()
        mutation_step = self.initial_mutation_step
        for _ in range(number_of_generations):
            new_population = self._process_generation(population, mutation_step)
            mutation_step = self.mutation_step_tuner.adapt_step(
                mutation_step, population, new_population, self.evaluation_function
            )
            population = new_population
        return population

    def _process_generation(self, population: np.ndarray, mutation_step: float):
        mutants = self._mutate(population, mutation_step)
        candidates = self._crossover(population, mutants)
        new_population = self._succession(population, candidates)
        return new_population

    def _mutate(self, population: np.ndarray, mutation_step: float):
        choosen_indices = np.random.choice(population.shape[0], population.shape[0])
        choosen = population[choosen_indices]
        points = np.array(
            [
                population[np.random.choice(population.shape[0], 2, replace=False)]
                for _ in range(population.shape[0])
            ]
        )
        substracted = points[:, 0, :] - points[:, 1, :]
        mutants = choosen + mutation_step * substracted
        return mutants

    def _crossover(self, population: np.ndarray, mutants: np.ndarray):
        guaranteed_crossovers_indices = np.random.choice(
            population.shape[1], population.shape[0]
        )
        crossovers_mask = np.random.choice(
            [True, False],
            population.shape,
            p=[self.crossover_rate, 1 - self.crossover_rate],
        )
        crossovers_mask[
            np.indices(guaranteed_crossovers_indices.shape)[0],
            guaranteed_crossovers_indices,
        ] = True
        candidates = np.where(crossovers_mask, mutants, population)
        return candidates

    def _succession(self, population: np.ndarray, candidates: np.ndarray):
        population_evaluations = self.evaluation_function(population)
        candidates_evaluations = self.evaluation_function(candidates)
        mask = candidates_evaluations <= population_evaluations
        new_population = np.where(mask[:, None], candidates, population)
        return new_population
