import numpy as np

from cec2017.functions import f1, f3, f4
from cec2017.utils import surface_plot
from src.differential_evolution import DifferentialEvolution
from src.mutation_step_tuners import MSRTuner, PSRTuner

if __name__ == "__main__":
    all_functions = [f4]
    samples = 100
    dimension = 10
    for function in all_functions:
        initial_population = np.random.uniform(-100, 100, size=(samples, dimension))
        basic_model = DifferentialEvolution(evaluation_function=function)

        print(function(initial_population))
        val = function(basic_model.evolve(initial_population, number_of_generations=500))
        for i in range(samples):
            print(f"{function.__name__}(x_{i}) = {val[i]:.6f}")

        median_model = DifferentialEvolution(
            evaluation_function=function, mutation_step_tuner=MSRTuner()
    )
        val = function(median_model.evolve(initial_population, number_of_generations=500))
        for i in range(samples):
            print(f"{function.__name__}(x_{i}) = {val[i]:.6f}")

        population_model = DifferentialEvolution(
            evaluation_function=function, mutation_step_tuner=PSRTuner()
    )
        val = function(population_model.evolve(initial_population, number_of_generations=500))
        for i in range(samples):
            print(f"{function.__name__}(x_{i}) = {val[i]:.6f}")
