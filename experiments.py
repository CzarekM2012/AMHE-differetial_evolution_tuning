import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cec2017.functions import all_functions

from src.differential_evolution import DifferentialEvolution
from src.mutation_step_tuners import MSRTuner, PSRTuner

POP_MULTIPLIER = 10
NUM_RUNS = 10
all_funcs = [all_functions[0]] + all_functions[2:]  # remove f2


def plot_series(
    n,
    series1,
    series2,
    series3,
    labels=("DE", "DE+MSR", "DE+PSR"),
    title="Title",
    xlabel="Generation",
    ylabel="Value",
    filename="results/plot.png",
):
    if len(series1) != n or len(series2) != n or len(series3) != n:
        raise ValueError("The length of series data must match the range 'n'")
    x = range(1, n + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(x, series1, label=labels[0])
    plt.plot(x, series2, label=labels[1])
    plt.plot(x, series3, label=labels[2])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def experiment_track(func: callable, dimension: int = 10, generations: int = 2000):
    pop_size = dimension * POP_MULTIPLIER
    basic_model = DifferentialEvolution(evaluation_function=func)
    msr_model = DifferentialEvolution(
        evaluation_function=func, mutation_step_tuner=MSRTuner()
    )
    psr_model = DifferentialEvolution(
        evaluation_function=func, mutation_step_tuner=PSRTuner()
    )

    initial_population = np.random.uniform(
        low=-100.0, high=100.0, size=(pop_size, dimension)
    )
    _, basic_stats = basic_model.evolve(
        initial_population, number_of_generations=generations, stats=True
    )
    _, msr_stats = msr_model.evolve(
        initial_population, number_of_generations=generations, stats=True
    )
    _, psr_stats = psr_model.evolve(
        initial_population, number_of_generations=generations, stats=True
    )
    plot_series(
        generations,
        basic_stats["bests"],
        msr_stats["bests"],
        psr_stats["bests"],
        title="Najlepsze rozwiązanie w kolejnych generacjach",
        xlabel="Generacja",
        ylabel="Najlepsza wartość rozwiązania",
        filename=f"results/best-D{dimension}-{func.__name__}.png",
    )
    plot_series(
        generations,
        basic_stats["means"],
        msr_stats["means"],
        psr_stats["means"],
        title="Średnia wartość rozwiązania w kolejnych generacjach",
        xlabel="Generacja",
        ylabel="Srednia wartość funkcji celu",
        filename=f"results/mean-D{dimension}-{func.__name__}.png",
    )
    plot_series(
        generations,
        basic_stats["stds"],
        msr_stats["stds"],
        psr_stats["stds"],
        title="Odchylenie standardowe w kolejnych generacjach",
        xlabel="Generacja",
        ylabel="Odchylenie standardowe",
        filename=f"results/std-D{dimension}-{func.__name__}.png",
    )


def experiment_all(dimension: int = 10, generations: int = 1000):
    pop_size = dimension * POP_MULTIPLIER

    stats = {
        "fnames": [],
        "basic_bests": [],
        "basic_means": [],
        "basic_stdvs": [],
        "basic_times": [],
        "MSR_bests": [],
        "MSR_means": [],
        "MSR_stdvs": [],
        "MSR_times": [],
        "PSR_bests": [],
        "PSR_means": [],
        "PSR_stdvs": [],
        "PSR_times": [],
    }

    for func in all_funcs:
        stats["fnames"].append(func.__name__)
        basic_bests = []
        basic_times = []
        msr_bests = []
        msr_times = []
        psr_bests = []
        psr_times = []

        basic_model = DifferentialEvolution(evaluation_function=func)
        msr_model = DifferentialEvolution(
            evaluation_function=func, mutation_step_tuner=MSRTuner()
        )
        psr_model = DifferentialEvolution(
            evaluation_function=func, mutation_step_tuner=PSRTuner()
        )

        for _ in range(NUM_RUNS):
            initial_population = np.random.uniform(
                low=-100.0, high=100.0, size=(pop_size, dimension)
            )
            start = time.time()
            result, _ = basic_model.evolve(
                initial_population, number_of_generations=generations
            )
            end = time.time()
            basic_times.append(end - start)
            basic_bests.append(np.nanmin(func(result)))

            start = time.time()
            result, _ = msr_model.evolve(
                initial_population, number_of_generations=generations
            )
            end = time.time()
            msr_times.append(end - start)
            msr_bests.append(np.nanmin(func(result)))

            start = time.time()
            result, _ = psr_model.evolve(
                initial_population, number_of_generations=generations
            )
            end = time.time()
            psr_times.append(end - start)
            psr_bests.append(np.nanmin(func(result)))
        collect_stats(
            stats, basic_bests, msr_bests, psr_bests, basic_times, msr_times, psr_times
        )
    return pd.DataFrame(stats)


def collect_stats(
    stats_dict: dict,
    basic_bests,
    msr_bests,
    psr_bests,
    basic_times,
    msr_times,
    psr_times,
):
    stats_dict["basic_bests"].append(np.nanmin(basic_bests))
    stats_dict["basic_means"].append(np.nanmean(basic_bests))
    stats_dict["basic_stdvs"].append(np.nanstd(basic_bests))
    stats_dict["basic_times"].append(np.nanmean(basic_times))

    stats_dict["MSR_bests"].append(np.nanmin(msr_bests))
    stats_dict["MSR_means"].append(np.nanmean(msr_bests))
    stats_dict["MSR_stdvs"].append(np.nanstd(msr_bests))
    stats_dict["MSR_times"].append(np.nanmean(msr_times))

    stats_dict["PSR_bests"].append(np.nanmin(psr_bests))
    stats_dict["PSR_means"].append(np.nanmean(psr_bests))
    stats_dict["PSR_stdvs"].append(np.nanstd(psr_bests))
    stats_dict["PSR_times"].append(np.nanmean(psr_times))


if __name__ == "__main__":
    np.random.seed(4321)
    for func in all_funcs:
        experiment_track(func, dimension=10, generations=1000)
        experiment_track(func, dimension=30, generations=1000)
    results = experiment_all(dimension=10, generations=1000)
    results.to_csv("results/results_D10_all.csv")
    results = experiment_all(dimension=30, generations=1000)
    results.to_csv("results/results_D30_all.csv")
