#!/usr/bin/env python3
"""
The iris.csv file contains 3 species of iris. This script will create linear regressions and linear regression plots for each
species separately using pandas and matplotlib. 

First, the dataset is read in from a csv file, then we perform linear regression of sepal vs. petal length by species, then
create one plot for each species.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load the iris dataset from iris.csv.

    Parameters
    ----------
    csv_path : str
        Path to CSV file

    Returns
    _______
    pd.DataFrame
        The iris dataset is returned as  a pandas DataFrame.
    """
    return pd.read_csv(csv_path)

def get_species_data(dataframe: pd.DataFrame, species_name : str) -> pd.DataFrame:
    """
    Return rows for single iris species only.

    Parameters
    __________
    dataframe : pd.DataFrame
        full iris dataset
    species_name : str
        species filtering for

    Returns
    _______
    pd.DataFrame
        filtered data frame containing only selected species
    """
    return dataframe[dataframe["species"] == species_name]

def run_regression(species_dataframe: pd.DataFrame):
    """
    run a linear regression of sepal vs. petal length

    Parameters
    __________
    species_dataframe : pd.DataFrame
        Data for one iris species

    Returns
    -------
    scipy.stats._stats_py.LinregressResult
        Regressions results object
    """
    x_values = species_dataframe["petal_length_cm"]
    y_values = species_dataframe["sepal_length_cm"]
    return stats.linregress(x_values, y_values)

def make_plot(species_dataframe: pd.DataFrame, species_name: str, output_dir: str = ".") -> None:
    """
    Create and save a scatterplot with a regression line for one species

    Parameters
    ----------
    species_dataframe : pd.DataFrame
        data for one iris species
    species_name : str
        species name used in the plot title and output filename
    output_dir : str, optional
        directory where the PNG file will be saved
    """
    x_values = species_dataframe["petal_length_cm"]
    y_values = species_dataframe["sepal_length_cm"]

    regression = run_regression(species_dataframe)
    slope = regression.slope
    intercept = regression.intercept

    plt.figure()
    plt.scatter(x_values, y_values, label="Data")
    plt.plot(x_values, slope * x_values + intercept, color="orange", label="Fitted line")
    plt.xlabel("Petal length (cm)")
    plt.ylabel("Sepal length (cm)")
    plt.title(species_name.replace("_", " "))
    plt.legend()

    output_path = Path(output_dir) / f"{species_name.lower()}_petal_vs_sepal_regression.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(
        f"{species_name}: slope={slope:.4f}, intercept={intercept:.4f}, "
        f"r={regression.rvalue:.4f}, p={regression.pvalue:.4e}"
    )
    print(f"Saved plot to {output_path}")


def main() -> None:
    """
    Load the Iris data and make one regression plot per species.
    """
    dataframe = load_data("iris.csv")

    species_list = dataframe["species"].unique()

    for species_name in species_list:
        species_dataframe = get_species_data(dataframe, species_name)
        make_plot(species_dataframe, species_name)


if __name__ == "__main__":
    main()
