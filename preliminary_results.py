import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os
    # import re
    # import random

    from tqdm import tqdm
    from tqdm.notebook import tqdm_notebook
    tqdm_notebook.pandas()
    return os, pd, plt, sns, tqdm


@app.cell
def _(os):
    project_root_path = os.path.join('.')
    experiment_path = os.path.join(project_root_path, 'experiments', 'preliminary')
    results_path = os.path.join(experiment_path, 'results')
    return (results_path,)


@app.cell
def _(os, pd, results_path):
    def load_results(csv_file):
        return pd.read_csv(csv_file, index_col=0)

    result_files = [file for file in os.listdir(results_path) if file.endswith('.csv')]
    print(f'Number of result files: {len(result_files)}')
    return (result_files,)


@app.cell
def _(os, pd, result_files, results_path, tqdm):
    # Initialize an empty DataFrame to store the combined data
    df = pd.DataFrame()

    # Read and combine all files
    for file in tqdm(result_files):
        file_path = os.path.join(results_path, file)
        curr_df = pd.read_csv(file_path, index_col=0)
        df = pd.concat([df, curr_df], ignore_index=True)

    df
    return (df,)


@app.cell
def _(df, plt, sns):
    # Create boxplot
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Number of anomalies', y='Metric time', hue='Metric', data=df)
    plt.title('Comparison of Execution Times by Metric')
    plt.xlabel('Metric')
    plt.ylabel('Execution Time')
    plt.show()
    return


@app.cell
def _(df, plt, sns):
    # Create boxplot
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Length', y='Metric time', hue='Metric', data=df)
    plt.title('Comparison of Execution Times by Metric')
    plt.xlabel('Metric')
    plt.ylabel('Execution Time')
    plt.show()
    return


@app.cell
def _(df, plt, sns):
    # Create boxplot
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Anomalies average length', y='Metric time', hue='Metric', data=df)
    plt.title('Comparison of Execution Times by Metric')
    plt.xlabel('Metric')
    plt.ylabel('Execution Time')
    plt.show()
    return


if __name__ == "__main__":
    app.run()
