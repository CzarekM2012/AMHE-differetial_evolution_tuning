import pandas as pd
from scipy.stats import wilcoxon


def read_file(filename: str):
    df = pd.read_csv(filename)
    df = df.dropna(axis=0)
    return df


def wilcoxon_results(df: pd.DataFrame):
    print(f'Basic bests vs MSR bests p-value: {round(wilcoxon(df["basic_bests"], df["MSR_bests"])[1], 4)}')
    print(f'Basic bests vs PSR bests p-value: {round(wilcoxon(df["basic_bests"], df["PSR_bests"])[1], 4)}')
    print(f'MSR bests vs PSR bests p-value: {round(wilcoxon(df["MSR_bests"], df["PSR_bests"])[1], 4)}')
    print(f'Basic times vs MSR times p-value: {round(wilcoxon(df["basic_times"], df["MSR_times"])[1], 4)}')
    print(f'Basic times vs PSR times p-value: {round(wilcoxon(df["basic_times"], df["PSR_times"])[1], 4)}')
    print(f'MSR times vs PSR times p-value: {round(wilcoxon(df["MSR_times"], df["PSR_times"])[1], 4)}')


if __name__ == "__main__":
    data_10 = read_file('results/results_D10_all.csv')
    print("Wilcoxon results, D=10:")
    wilcoxon_results(data_10)
    print("\n")
    data_30 = read_file('results/results_D30_all.csv')
    print("Wilcoxon results, D=30:")
    wilcoxon_results(data_30)
