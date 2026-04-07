#!/usr/bin/env python3
"""
The iris.csv file contains 3 species of iris. This script will create linear regressions and linear regression plots for each
species separately using pandas and matplotlib. 

First, the dataset is read in from a csv file, then we perform linear regression of sepal vs. petal length by species, then
create one plot for each species.
"""

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

dataframe = pd.read_csv("iris.csv")
long_flowers = dataframe[dataframe.petal_length_cm > 5.9]

versicolor = dataframe[dataframe.species == "Iris_versicolor"]
virginica = dataframe[dataframe.species == "Iris_virginica"]
setosa = dataframe[dataframe.species == "Iris_setosa"]

# Regression: Overall
x = dataframe.petal_length_cm
y = dataframe.sepal_length_cm
regression = stats.linregress(x, y)
slope = regression.slope
intercept = regression.intercept
plt.scatter(x, y, label = 'Data')
plt.plot(x, slope * x + intercept, color = "orange", label = 'Fitted line')
plt.xlabel("Petal length (cm)")
plt.ylabel("Sepal length (cm)")
plt.legend()
plt.savefig("petal_v_sepal_length_regress.png")
plt.close()

# Regression: versicolor
x = versicolor.petal_length_cm
y = versicolor.sepal_length_cm
regression = stats.linregress(x, y)
slope = regression.slope
intercept = regression.intercept
plt.scatter(x, y, label = 'Data')
plt.plot(x, slope * x + intercept, color = "orange", label = 'Fitted line')
plt.xlabel("Petal length (cm)")
plt.ylabel("Sepal length (cm)")
plt.legend()
plt.savefig("versicolor_petal_v_sepal_length_regress.png")
plt.close()

# Regression: virginica
x = virginica.petal_length_cm
y = virginica.sepal_length_cm
regression = stats.linregress(x, y)
slope = regression.slope
intercept = regression.intercept
plt.scatter(x, y, label = 'Data')
plt.plot(x, slope * x + intercept, color = "orange", label = 'Fitted line')
plt.xlabel("Petal length (cm)")
plt.ylabel("Sepal length (cm)")
plt.legend()
plt.savefig("virginica_petal_v_sepal_length_regress.png")
plt.close()

# Regression: setosa
x = setosa.petal_length_cm
y = setosa.sepal_length_cm
regression = stats.linregress(x, y)
slope = regression.slope
intercept = regression.intercept
plt.scatter(x, y, label = 'Data')
plt.plot(x, slope * x + intercept, color = "orange", label = 'Fitted line')
plt.xlabel("Petal length (cm)")
plt.ylabel("Sepal length (cm)")
plt.legend()
plt.savefig("setosa_petal_v_sepal_length_regress.png")
plt.close()
