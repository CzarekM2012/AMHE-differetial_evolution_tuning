import numpy as np

from src.differential_evolution import DifferentialEvolution

if __name__ == "__main__":

    def evaluation(matrix):
        return np.sum(matrix**2, axis=1)

    initial_population = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [9.0, 4.0, 3.0, 7.0],
            [7.0, 6.0, 3.0, 4.0],
            [6.0, 1.0, 7.0, 3.0],
            [0.0, 3.0, 7.0, 8.0],
            [5.0, 3.0, 7.0, 6.0],
            [7.0, 1.0, 3.0, 8.0],
            [2.0, 0.0, 4.0, 3.0],
            [3.0, 1.0, 9.0, 0.0],
            [4.0, 8.0, 6.0, 2.0],
        ]
    )

    model = DifferentialEvolution(evaluation_function=evaluation)
    print(model.evolve(initial_population))
