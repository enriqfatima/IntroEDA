#!/usr/bin/env python
# coding: utf-8

# In[8]:


# import pandas & numpy 
import pandas as pd
import numpy as np

#import seaborn
import seaborn as sns

# import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style as style
# set plotting size parameter
plt.rcParams['figure.figsize'] = (12, 5)


# Input the data from the repo in as a dataframe: 

# In[9]:


df = pd.read_csv('https://raw.githubusercontent.com/fivethirtyeight/data/master/candy-power-ranking/candy-data.csv')
df.shape


# Data Summary

# In[10]:


candy_shape = df.shape
candy_shape


# This allows you to get a sense of the information stored in your dataset overall.

# In[11]:


df


# A full description of these data can be found [here](https://github.com/fivethirtyeight/data/tree/master/candy-power-ranking). From that link, we'll include a description of each variable (column) here:
# 
# Header | Description
# -------|------------
# chocolate | Does it contain chocolate?
# fruity | Is it fruit flavored?
# caramel | Is there caramel in the candy?
# peanutalmondy | Does it contain peanuts, peanut butter or almonds?
# nougat | Does it contain nougat?
# crispedricewafer | Does it contain crisped rice, wafers, or a cookie component?
# hard | Is it a hard candy?
# bar | Is it a candy bar?
# pluribus | Is it one of many candies in a bag or box?
# sugarpercent | The percentile of sugar it falls under within the data set.
# pricepercent | The unit price percentile compared to the rest of the set.
# winpercent | The overall win percentage according to 269,000 matchups.
# 

# Use the `describe` method to calculate and display these summary statistics: 

# In[12]:


df = df.value_counts(normalize = False).rename('proportion').reset_index()
df


# Generate a histogram of the `winpercent` column with 15 bins.

# In[13]:


plt.rcParams['figure.figsize'] = (16,10) #default plot size to output
sns.histplot(df['winpercent'], bins=15, kde=False);

f1 = plt.gcf()


# Using the `value_counts()` method, I can determine how many different possible values there are for the `chocolate` series in the `df` DataFrame and how many observations fall into each. 

# In[14]:


chocolate_values = df['chocolate'].value_counts()
chocolate_values


# There are a number of different ways in which I can determine whether or not data are missing. The most common approaches are summarized here:
# 
# ```python
# # Calculate % of missing values in each column:
# df.isna().mean()
# 
# # Drop columns with any missing values:
# df.dropna(axis='columns')
# 
# # Drop columns in which more than 10% of values are missing:
# df.dropna(thresh=len(df)*0.9, axis='columns')
# 
# # Want to know the *count* of missing values in a DataFrame?
# df.isna().sum().sum()
# 
# # Just want to know if there are *any* missing values?
# df.isna().any().any()
# df.isna().any(axis=None)
# ```
# 
# Run the following cell and interpret the ouput:

# In[15]:


df.isna().any()


# I can determine how many variables have missing data in this dataset: 

# In[16]:


var_missing = df.isna().sum().sum()
var_missing 


# To demonstrate this, if I wanted to replace the zeroes and ones in the current dataset with boolean values, true or false, I could do this using `replace`.

# In[17]:


df_bool = df.replace({0:False, 1:True})
df_bool


# I can generate a barplot displaying the number of True and False values for the `chocolate` series.

# In[18]:


f2 = sns.countplot(x='chocolate', data=df_bool, color = 'brown')
f2 = plt.gcf()


# To just get the data I am interested in, I can return a DataFrame with the columns: `competitorname`, `chocolate`, `fruity`, `hard`, `bar`, `pluribus`, `sugarpercent`, `pricepercent`, and `winpercent`.
# 
# I stored this output  in `df` (overwriting the previous data stored in `df`).

# In[19]:


df = df[['competitorname', 'chocolate', 'fruity', 'hard', 'bar', 'pluribus', 'sugarpercent', 'pricepercent', 'winpercent']]
df.shape


# To remind myself of what type of information is stored in each column, the `dtypes` attribute can be very helpful:

# In[20]:


df.dtypes


# In[21]:


df_bool.dtypes


# For the `df_bool` dataset, I am interested in selecting only the columns that are either a string (object) or a bool: 

# In[22]:


df_bool = df_bool.select_dtypes(include=['bool'or'object'] )
df_bool.shape


# Generally I know that there are three approaches to renaming columns:
# 1. Most flexible option: `df = df.rename({'A':'a', 'B':'b'}, axis='columns')`
# 2. Overwrite all column names: `df.columns = ['a', 'b']`
# 3. Apply string method: `df.columns = df.columns.str.lower()`
# 
# Below, I have `renamed` `pluribus` to `multicandy_pack` for both the `df` and `df_bool` dataframes.

# In[23]:


df = df.rename({'competitorname': 'competitorname', 'chocolate': 'chocolate' , 'fruity':'fruity', 'hard':'hard', 'bar':'bar', 'pluribus':'multicandy_pack', 'sugarpercent':'sugarpercent', 'pricepercent':'pricepercent', 'winpercent':'winpercent'}, axis='columns')
df_bool = df_bool.rename({'competitorname': 'competitorname', 'chocolate': 'chocolate' , 'fruity':'fruity', 'hard':'hard', 'bar':'bar', 'pluribus':'multicandy_pack', 'sugarpercent':'sugarpercent', 'pricepercent':'pricepercent', 'winpercent':'winpercent'}, axis='columns')
df.columns
df_bool.columns


# Often with data we also need to add new columns; thus to do this, I typically use one of two approaches, summarized generally here: 
# 1. `assign`
# ```python
# df.assign(new_col = df['col'] * val)
# ```
# 2. `apply`
# ```python
# for col in df.columns:
#       df[f'{col}_new'] = df[col].apply(my_function)
# ```
# 
# Below I used `assign` to add a new column to `df` called `fruity_choco` that adds the value in `chocolate` to the value in `fruity`. This way, the value will be `0` if it is neither fruity nor chocolate, `1` if it is one or the other, and `2` if it is both.

# In[24]:


df = df.assign(fruity_choco = df['chocolate']+ df['fruity'])
df


# In[25]:


df['fruity_choco'].value_counts()


# Using `value_counts()` above on the `fruity_choco` column I created, I can see that there is one candy that is both chocolate and fruity in our dataset. 

# In[26]:


both = df.loc[df['fruity_choco'] == 2]
both


# Using the filter concept, I want to have my dataframe to only include rows that contain candy that is fruity or chocolate in some capacity (meaning, something that is both fruity and chocolate *would* be included). 

# In[27]:


df = df[df['fruity_choco'].isin([1,2])]
df


# I know that calculations can be carried out on columns using conditionals. Thus, I can determine how many of the candies in my current dataset are part of a pack of candy with multiple types of candy in the pack. To do this, I would use `sum()` on the subset of the data that meets that condition

# In[28]:


(df['multicandy_pack']==1).sum()


# At the moment, my dataframe is currently sorted in alphabetical order by competitor name (candy). Below, I sorted the dataframe to be ascending by surgar percentile.

# In[29]:


df = df.sort_values(by='sugarpercent', ascending = False).reset_index()
df


# I can `groupby` a candy's `chocolate` status and then determine the average value for the other variables in the dataset.

# In[30]:


df.drop(columns='competitorname').groupby('chocolate').agg('mean')


# Below I carried out a similar operation, group `df` by `fruity`, and then calculated the minimum, average (mean), and maximum values, for the `sugarpercent` column.

# In[31]:


sugar_summary = df.groupby('fruity')['sugarpercent'].agg(['min','max','mean'])
sugar_summary

