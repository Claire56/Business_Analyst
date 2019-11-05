# Normalize data (length of 1)
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
#import seaborn as sns
from pandas import read_csv
#from numpy import set_printoptions
from pandas import set_option
import pandas as pd
import numpy as np

filename = 'DiabetesDataSet.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

df = read_csv(filename, names=names)
array = df.values

# save features as pandas dataframe 
X1 = df.drop(df.columns[8], axis = 1)
Y1 = df.drop(df.columns[0:8], axis = 1)

# separate array into input and output components
X = array[:,0:8]
Y = array[:,8]

'''
Let's check to see if there are any nulls/Nan in the dataset
Checking for missing values and Not-a-number(Nan) entries 

'''
df.shape
sumNullRws = df.isnull().sum()
# remove null elements in data
df = df.dropna()
# check to see if there is any nulls left
df.isnull().sum()



# normalize the x data only
scaler = Normalizer().fit(X)
normalizedX = scaler.transform(X)
## summarize transformed data
#set_printoptions(precision=3)
#print(normalizedX[0:5,:])


# First perform exploratory data analysis using correlation and scatter plot
# look at the first 20 rows of data
peek = df.head(20)
print(peek)

# descriptive statistics: mean, max, min, count, 25 percentile, 50 percentile, 75 percentile
set_option('display.width', 100)
set_option('precision', 1)
description = df.describe()
print(description)

# we look at the distribution of data and its descriptive statistics
df.hist()
plt.show()

# build a new dataframe with normalized data
name = names[0:len(names)-1]
df2 = pd.DataFrame(normalizedX,columns = name)
df_norm = pd.concat([df2,Y1], axis = 1)

set_option('display.width', 100)
set_option('precision', 1)
descpt_norm = df_norm.describe()
print(descpt_norm)

# plot histogram of normalized data
df_norm.hist()
plt.show()


# now standardize the data
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
## summarize transformed data
#set_printoptions(precision=3)
#print(rescaledX[0:5,:])
# build a new dataframe with normalized data

df3 = pd.DataFrame(rescaledX,columns = name)
df_rescaled = pd.concat([df3,Y1], axis = 1)

set_option('display.width', 100)
set_option('precision', 1)
descpt_rescaled = df_rescaled.describe()
print(descpt_rescaled)

# plot histogram of normalized data
descpt_rescaled.hist()
plt.show()

'''
Perform log transformation on one column of the data. We normally perform log
transformation when we want to correct the skewness of the distribution. Log transformation
will change the distribution closer to normal.
'''
df.shape

array = np.log(df['age'].values)

#dataset1.assign(Salary = pd.Se(array))
df.loc[:,'ageLog'] = pd.Series(array, index=df.index)
df.describe()  

df = df.dropna()

df.head(20)
colTags = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'ageLog']
#X = dataset.loc[:,['Hits', 'Years', 'RBI','Walks', 'Runs', 'PutOuts']]
X_txm = df.loc[:,colTags]
#X = dataset.drop('Petrol_Consumption', axis=1)  
y_txm = df['class'] 
#dataset1 = pd.DataFrame() 
df_txm = pd.concat([X_txm, y_txm], axis=1)
# First perform exploratory data analysis using correlation and scatter plot
# look at the first 20 rows of data
peek = df_txm.head(20)
print(peek)

# descriptive statistics: mean, max, min, count, 25 percentile, 50 percentile, 75 percentile
set_option('display.width', 100)
set_option('precision', 1)
description_txm = df_txm.describe()
print(description_txm)

# we look at the distribution of data and its descriptive statistics
plt.figure() # new plot
df_txm.hist()
plt.show()