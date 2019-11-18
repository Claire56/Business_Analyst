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

filename = 'boston.csv'
data1 = read_csv(filename)
array = data1.values
# separate array into input and output components (Numpy arrays)
X = array[:,0:14]
Y = array[:,14]

## save features as pandas dataframe for stepwise feature selection
X1 = data1.drop(data1.columns[14], axis = 1)
Y1 = data1.drop(data1.columns[0:14], axis = 1)

scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
# summarize transformed data
set_printoptions(precision=10)
print(rescaledX[0:5,:])

scaler1 = StandardScaler().fit(X1)
rescaledX1 = scaler.transform(X1)

# you can make a new data frame with the standardized data
dataStandDf = pd.DataFrame(rescaledX)
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
#### COMMENT: the function helps to show all the statistics figure of the dataframe, however, understanding the number will require the visualizations of these statistics##

# we look at the distribution of data and its descriptive statistics
plt.figure() # new plot
data1.hist()
plt.show()
###COMMENT: looking at the histogram, we can tell what is the most frequently showed data, for example, for data of age and black, most of the observations get the value at the most right of the range, which indicates the mean of these two data column must be dragged a little bit to the right; meanhile, medv and rm have a quite more like a bell-shape distribution - standard normal distribution; lstat and dis have most of the observation distributed at the most left hand side of the range, which means the mean is to the left of the histogram##

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
####COMMENT: looking at the heatmap, I can see that the dependent variable (lstat) has a quite strong negative correlation with zn, rm, dis,medv and a positive correlation with tax and rad; as for other pair-wise correlation, the lighter the color is the more positively strong the relationship is and the darker the color is the more negatively strong relationship is##
####COMMENT: if there is strong correlation between two variables, the regression model including the two variables would give a high goodness of fit statistics (R-squared)###
# scatter plot of all data
plt.figure()
scatter_matrix(data1)
plt.show()

####COMMENT: the correlation matrix base on the assumption that pair-wise correlation follows a linear shape, in case correlation heatmap shows there is no linear relationship between two particular variables, there might still be a non-linear correlation, the scatter plot diagram can show clearer this aspects### 
