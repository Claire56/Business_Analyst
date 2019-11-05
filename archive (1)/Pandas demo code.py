# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 06:55:59 2018

@author: jahan
"""

import pandas as pd

# Pandas data series
purchase1 = pd.Series({'Name': 'Chris',
                        'Item Purchased' : 'Dog Food',
                        'Cost' : 22.50})
purchase2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased' : 'Kitty Litter',
                        'Cost' : 2.50})
purchase3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased' : 'Bird Seed',
                        'Cost' : 5.00})
# Make a dataframe out of dataSeries
df = pd.DataFrame([purchase1, purchase2, purchase3])
df = pd.DataFrame([purchase1, purchase2, purchase3], index = ['Store1', 'Store1', 'Store2'])

# Display data head
df.head(2)
# display data by row tag
df.loc['Store2']
# get type
type(df.loc['Store2'])
# select row and column
df.loc['Store1', 'Cost']

# transpose dataframe
df1 = df.T
# Get cost as a row tag
df.T.loc['Cost']
# to get only a column you don't need the .loc method
df['Cost']
df.loc['Store1']['Cost']
# Retrieving Column Using Slicing

df.loc[:,['Name', 'Cost']]

# to drop a column, you need to specify the axis, axis=0 is row and axis=1 is column
df2 = df.drop(['Cost'], axis=1)
# to get help on a method
#df.drop?

del df['Cost']
# to add a new empty column with tag new_col or a column of zeros
df['new_col1'] = ""
df['new_col2'] = 1
df['new_col3'] = 2
print(df)

# adding columns by transforming existing columns
df['new_col4'] = df[['new_col2','new_col3']].sum(axis=1)
# adding a new column in a different way
old_col_list = ['new_col3','new_col4']
df['new_col5'] = df[old_col_list].apply(sum, axis=1)
print(df)
# combining dataframes
# generate dataframes
df1 = pd.DataFrame({
        'col3' : ['x', 'y', 'z'],
        'new_col1' : "",
        'new_col2' : [1, 2, 3],
        'new_col3' : [7, 8, 9],
#        'col4'     : [7, 8, 9],        
        'new_col4' : [10, 11, 12],
        'new_col5' : [13, 14, 15],
        'new_col6' : [16, 17, 18]
        })
print(df1)
# second dataframe
df2 = pd.DataFrame({
    'col3': ['a', 'b', 'c', 'd'],
    'new_col1': '',
    'new_col2': 0,
    'new_col3': [11, 13, 15, 17],
    'new_col4': [17, 19, 21, 23],
    'new_col5': [7.5, 8.5, 9.5, 10.5],
    'new_col6': [13, 14, 15, 16]
})

print(df2)

# now combine the two dataframes
df3 = pd.concat([df1, df2], ignore_index = True)
print(df3)

# convert a column to a list
my_list = df3['new_col6'].tolist()
# read a .csv file without any processing
df_olympic =pd.read_csv('olympic.csv', encoding='windows-1252')
#df5 =pd.read_csv('DiabetesDataSet')
# read with using first column as index and skip the first row
df5 = pd.read_csv('olympic.csv', index_col = 1, skiprows = 1, encoding='windows-1252')
df5.head(10)
df5 = df5.drop(['Unnamed: 0'], axis=1)
# list the column tags
df5.columns

# change labels to Gold, silver, Bronze
for col in df5.columns:
    if col[:2] == '01':
        #print(col[4:0])
        df5.rename(columns={col:'Gold' + col[4:]}, inplace=True)
    if col[:2] == '02':
        df5.rename(columns={col:'Silver' + col[4:]}, inplace=True)
    if col[:2] == '03':
        df5.rename(columns={col:'Bronze' + col[4:]}, inplace=True)
#    if col[:2] == '04':
#        df5.rename(columns={col:'Silver' + col[4:0]}, inplace=True)  
    
    
df5.head(5) 
        