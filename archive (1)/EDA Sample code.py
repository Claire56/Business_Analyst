# -*- coding: utf-8 -*-
"""
Homework 2 Solution (Linear Regression and Exploratory data analysis)
Created on Mon Mar 12 00:53:30 2018
@author: jahan
Tested with Python 2.7
updated: 7/21/2018 to work with Python 3.6
Reference for the sample code on step-wise forward model: https://datascience.stackexchange.com/questions/937/does-scikit-learn-have-forward-selection-stepwise-regression-algorithm
"""


import pandas as pd
#import statsmodels.api as sm
#from sklearn.linear_model import LinearRegression
#from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from pandas import set_option
from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from numpy import set_printoptions

filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data1 = read_csv(filename, names=names)
array = data1.values
# separate array into input and output components (Numpy arrays)
X = array[:,0:8]
Y = array[:,8]

## save features as pandas dataframe for stepwise feature selection
X1 = data1.drop(data1.columns[8], axis = 1)
Y1 = data1.drop(data1.columns[0:8], axis = 1)

scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
# summarize transformed data
set_printoptions(precision=3)
print(rescaledX[0:5,:])

scaler1 = StandardScaler().fit(X1)
rescaledX1 = scaler.transform(X1)

# you can make a new data frame with the standardized data
dataStandDf = pd.DataFrame(rescaledX, columns = names[0:8])
dataStandDf['class'] = Y 

# First perform exploratory data analysis using correlation and scatter plot
# look at the first 20 rows of data
peek = data1.head(20)
print(peek)

# descriptive statistics: mean, max, min, count, 25 percentile, 50 percentile, 75 percentile
set_option('display.width', 100)
set_option('precision', 1)
description = data1.describe()
print(description)

# show descriptive stats after standardization
set_option('display.width', 100)
set_option('precision', 1)
description1 = dataStandDf.describe()
print(description1)

# we look at the distribution of data and its descriptive statistics
plt.figure() # new plot
data1.hist()
plt.show()


# correlation heat map, pay attention to correlation between all predicators/features and each predictor and the output
plt.figure() # new plot
corMat = data1.corr(method='pearson')
print(corMat)
# plot correlation matrix as a heat map
sns.heatmap(corMat, square=True)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.title("CORELATION MATTRIX USING HEAT MAP")
plt.show()

# scatter plot of all data
plt.figure()
scatter_matrix(data1)
plt.show()

