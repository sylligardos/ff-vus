"""
@who: Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: PhD Candidate, 2nd year (2025)
@what: FF-VUS
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class SylliGraph:
    def __init__(self, experiment_path):
        """
        Initialize the plotting class with defaults you can reuse forever.
        """
        self.save_fig_path = os.path.join(experiment_path, 'figures')
        os.makedirs(self.save_fig_path, exist_ok=True)

        # Consistent color palette
        self.color_palette = {
            'FF-VUS (L: 128)': '#eee13f',          # Good
            'FF-VUS-GPU (L: 128)': '#40da70',     # Very Good
            'AUC': '#f1a73f',                # Secondary competitor
            'VUS (L: 128)': '#b02a41',            # Main competitor
            'AFFILIATION': '#A5D1C2',        # Useless competitors
            'Range-AUC (L: 128)': '#547C6E',       # Useless competitors
            'RF': '#123327',                    # Useless competitors
        }

        # Map shorthand names to formal names
        self.formal_names = {
            'FF-VUS': 'FF-VUS (L: 128)',
            'FF-VUS-GPU': 'FF-VUS-GPU (L: 128)',
            'AUC': 'AUC',
            'VUS': 'VUS (L: 128)',
            'RF': 'RF',
            'AFFILIATION': 'AFFILIATION',
            'RANGE-AUC': 'Range-AUC (L: 128)',
        }

        # Default seaborn style
        sns.set_style("whitegrid")

    def _format_plot(self, title=None, xlabel=None, ylabel=None, rotate_xticks=True, axis=None):
        """
        Helper function to apply consistent styling to plots.
        """
        if axis is None:
            if title is not None:
                plt.title(title, fontsize=15)
            if xlabel is not None:
                plt.xlabel(xlabel, fontsize=14)
            if ylabel is not None:
                plt.ylabel(ylabel, fontsize=14)
            if rotate_xticks:
                plt.xticks(rotation=15, fontsize=11)
        else:
            if title is not None:
                axis.set_title(title, fontsize=15)
            if xlabel is not None:
                axis.set_xlabel(xlabel, fontsize=14)
            if ylabel is not None:
                axis.set_ylabel(ylabel, fontsize=14)
            if rotate_xticks:
                axis.tick_params(axis='x', rotation=15, labelsize=11)

    def _finalize_plot(self, filename):
        """
        Common utility to save and show the plot, avoiding code repetition.
        
        Args:
            filename (str or None): Base filename (without extension) to save the plot.
            description (str): Text description for logging.
        """
        if filename is not None:
            save_path = os.path.join(self.save_fig_path, filename)
            plt.savefig(f"{save_path}.svg", bbox_inches='tight')
            plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
            # print(f"Plot saved to {save_path}")
        
        plt.tight_layout()
        plt.show()

    def boxplot_exec_time(self, df, order=None, filename=None, title=None, xlabel=None, ylabel=None):
        """
        Create a reusable boxplot for comparing execution times across metrics.
        """
        plt.figure(figsize=(8, 5))

        # df['Metric'] = df['Metric'].map(self.formal_names).fillna(df['Metric'])
        df = df.replace(self.formal_names)

        medianprops = dict(linestyle='-', linewidth=2, color='white')
        meanprops = dict(
            marker='X',
            markerfacecolor='white',
            markeredgecolor='black',
            markersize=6
        )

        axis = sns.boxplot(
            y='Metric time',
            x='Metric',
            hue='Metric',
            data=df,
            showfliers=False,
            log_scale=True,
            showmeans=False,
            meanprops=meanprops,
            medianprops=medianprops,
            palette=self.color_palette,
            order=order,
            saturation=1,
        )
        # sns.violinplot(y='Metric time', x='Metric', hue='Metric', data=df, log_scale=True, palette=self.color_palette, order=order, saturation=1)

        self._format_plot(
            title=title,
            xlabel=xlabel,
            ylabel=ylabel
        )
        
        # Remove borders around the boxes but keep whiskers and caps
        for j, patch in enumerate(axis.patches):
            facecolor = patch.get_facecolor()
            if all([x == 1 for x in facecolor]):
                patch.set_facecolor((1, 1, 1, 0.0))  # alpha = 0.0
                patch.set_edgecolor((0, 0, 0, 1.0))  # fully opaque border
                patch.set_linewidth(1.5)

                axis.lines[j * 6 + 4].set_color('black')    # 6 lines per box and usually the median is the 4th
                axis.lines[j * 6 + 4].set_linewidth(1.5)
            else:
                patch.set_edgecolor(facecolor)
                patch.set_linewidth(0.5)

        # Save the figure
        plt.grid(axis='x')
        self._finalize_plot(filename)
        

    def boxplot_error(self, df, order=None, filename=None, title=None, xlabel=None, ylabel=None):
        plt.figure(figsize=(5, 3))

        df = df.replace(self.formal_names)
        
        sns.boxplot(
            df, 
            showfliers=True, 
            fill=True, 
            flierprops={"marker": "."}, 
            width=.5, 
            palette=self.color_palette, 
            saturation=1
        )

        plt.yscale('log')
        # plt.yticks([10**-x for x in range(0, 16)][:2:])
        

        self._format_plot(
            title=title,
            xlabel=xlabel,
            ylabel=ylabel
        )

        # Save the figure
        if filename is not None:
            save_path = os.path.join(self.save_fig_path, filename)
            plt.savefig(f"{save_path}.svg", bbox_inches='tight')
            plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
        plt.show()


    def plot_exec_time_per_attr(self, analysis_dfs, bins=20, filename='exec_time_per_attr.svg'):
        """
        Create a multi-subplot figure showing execution time per attribute
        for different evaluation metrics.

        Args:
            analysis_dfs (list): List of 3 dataframes [length_df, n_anomalies_df, avg_anom_len_df]
            bins (int): Number of bins for grouping if needed
            filename (str): Name of the file to save the figure
        """
        attributes = ["Length", "Number of anomalies", "Anomalies average length"]

        # Define subplot grid layout
        attr_axis = {
            "Length": (2, 1, 1), 
            "Number of anomalies": (2, 2, 3), 
            "Anomalies average length": (2, 2, 4),
        }

        fig = plt.figure(figsize=(10, 6))
        legend_axis = None

        for i, attribute in enumerate(attributes):
            axis = fig.add_subplot(*attr_axis[attribute])
            curr_df = analysis_dfs[i]

            # Main plot
            sns.lineplot(
                x=attribute,
                y='Metric time',
                hue='Metric',
                data=curr_df,
                ax=axis,
                palette=self.color_palette,
                markers=True,
                style='Metric',
                dashes=False,
                linewidth=2,
            )

            # Axis formatting
            self._format_plot(
                title=None,
                xlabel=attribute,
                ylabel="Execution time (seconds)" if i != 2 else "",
                rotate_xticks=False
            )
            axis.set_yscale('log')
            axis.grid(axis='both', alpha=0.5)

            if attribute == "Length":
                axis.set_xscale('log', base=2)

            # Keep legend only for the first plot
            if i != 0:
                axis.get_legend().remove()
            else:
                legend_axis = axis

        # --- Shared legend ---
        if legend_axis:
            handles, labels = legend_axis.get_legend_handles_labels()
            legend_axis.get_legend().remove()
            fig.legend(
                handles,
                labels,
                loc='upper center',
                bbox_to_anchor=(0.5, 1.02),
                ncol=len(labels),
                frameon=False,
                fontsize='small'
            )
        self._finalize_plot(filename)

    
    def plot_dataset_insights(self, curr_df, dataset_name=None):
        """Plot dataset insights"""
        metrics = curr_df['Metric'].unique()
        single_metric_df = curr_df[curr_df['Metric'] == metrics[0]]
        print(f"Total number of time series: {len(single_metric_df)}")
        print(f"Total number of points: {single_metric_df['Length'].sum()}, {single_metric_df['Length'].sum()//10**3}k, {single_metric_df['Length'].sum()//10**6}m, {single_metric_df['Length'].sum()//10**9}b")

        attributes = ["Length", "Number of anomalies", "Anomalies average length"]
        fig, ax = plt.subplots(1, 3, figsize=(15, 3))
        bins = 20

        for attr, axis in zip(attributes, ax):
            print(f"{attr} -> min: {curr_df[attr].min()}, max: {curr_df[attr].max()}")
            sns.histplot(x=attr, data=curr_df, ax=axis, bins=bins)
            axis.set_xlabel(attr)

        # plt.suptitle("Dataset insights" + f": {dataset_name}" if dataset_name is not None else "")
        self._finalize_plot(filename="dataset_insights" + f"_{dataset_name}" if dataset_name is not None else "")


    def plot_time_analysis_comparison(self, df, filename=None):
        """
        Creates a two-panel figure showing the runtime breakdown of
        different steps for FF-VUS (CPU) and FF-VUS-GPU.

        Args:
            df (pd.DataFrame): DataFrame containing metrics and timing info.
            filename (str, optional): If provided, saves the figure under this name.
        """
        # Extract relevant columns
        time_analysis_cols = [
            x for x in df.columns if 'time' in x and x != 'Metric time'
        ]
        metrics = ['FF-VUS (L: 128)', 'FF-VUS-GPU (L: 128)']

        # Filter DataFrame for relevant metrics
        curr_df = df[df['Metric'].isin(metrics)]

        # Create subplots
        fig, ax = plt.subplots(2, 1, figsize=(14, 6), sharey=False)

        # Loop through metrics and plot
        for j, metric in enumerate(metrics):
            metric_df = curr_df[curr_df['Metric'] == metric]
            sns.boxplot(
                data=metric_df[time_analysis_cols],
                ax=ax[j],
                palette=[self.color_palette.get(metric, "#333333")],
                showfliers=False
            )

            # Configure axis
            ax[j].set_xticks(np.arange(len(time_analysis_cols)))
            ax[j].set_xticklabels(time_analysis_cols, rotation=45, ha='right')
            ax[j].set_title(metric)
            ax[j].set_ylabel('Runtime (sec, log scale)')
            ax[j].set_yscale('log')
            ax[j].grid(axis='y', alpha=0.5)

        plt.tight_layout()
        self._finalize_plot(filename)