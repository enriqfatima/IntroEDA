#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports 
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()
sns.set_context('talk')

import warnings
warnings.filterwarnings('ignore')

import patsy
import statsmodels.api as sm
import scipy.stats as stats
from scipy.stats import ttest_ind, chisquare, normaltest


# ## Setup
# 
# Data: the responses collected from a previous survery of the COGS 108 class. 
# - There are 416 observations in the data, covering 10 different 'features'.
# 
# Research Question: Do students in different majors have different heights?
# 
# Background: Physical height has previously shown to correlate with career choice, and career success. More recently it has been demonstrated that these correlations can actually be explained by height in high school, as opposed to height in adulthood (1). It is currently unclear whether height correlates with choice of major in university. 
# 
# Reference: 1) https://www.sas.upenn.edu/~apostlew/paper/pdf/short.pdf
# 
# Hypothesis: We hypothesize that there will be a relation between height and chosen major. 

# In[2]:


# your code here
df = pd.read_csv('COGS108_IntroQuestionnaireData.csv')
df.shape


# Let's checkout the data: 

# In[3]:


df.head(5)


# In[4]:


# Renaming the columns of the dataframe
df.columns = ['timestamp', 'year', 'major', 'age', 'gender', 'height',
              'weight', 'eye_color', 'born_in_CA', 'favorite_icecream']


# In[5]:


df.isnull().head(5)


# In[6]:


rows_to_drop = df[df.isnull().any(axis=1)]
rows_to_drop.shape


# In[7]:


rows_to_drop


# In[8]:


df = df.dropna(subset=['major', 'height', 'gender', 'age']) #how = 'any', inplace = True)
df.shape


# In[9]:


def example_standardize_function(str_in):
    '''Standardize data to the question 'what is your favorite major python version?'
    
    Parameters
    ----------
    str_in : string
        A provided answer.
        
    Returns
    -------
    int_out : int or np.nan
        A standardized integer response.
    '''
    
    # Make the input all lowercase
    str_in = str_in.lower()
    
    # Drop all whitespace
    str_in = str_in.strip()
    
    # Replace things (and then strip again afterwords)
    # Note that the 'replace' replaces the first argument, with the second
    # The first argument does not need to be present in the string,
    # if it's not there 'replace' does nothing (but does not error), so the code moves on.
    str_in = str_in.replace('version', '')
    str_in = str_in.replace('python', '')
    str_in = str_in.strip()
    
    # Cast to integer, if what's left seems appropriate
    if str_in.isnumeric() and len(str_in) == 1:
        out = int(str_in)
    # Otherwise, consider input was probably ill-formed, return nan
    else: 
        out = np.nan
    
    return out

# Check how this function help standardize data:
# Example possible answers to the question 'What is your favourite major version of Python':
print('INPUT', '\t\t-\t', 'OUTPUT')
for inp in ['version 3', '42', '2', 'python 3', 'nonsense-lolz']:
    print('{:10s} \t-\t {:1.0f}'.format(inp, example_standardize_function(inp)))


# Below, we can observe the majors: 

# In[10]:


df['major'].unique()


# In[11]:


def standardize_major(string):
    
    string = string.lower()
    string = string.strip()
    
    if 'cog' in string:
        output = 'COGSCI'
    elif 'computer' in string:
        output = 'COMPSCI'
    elif 'cs' in string:
        output = 'COMPSCI'
    elif 'math' in string:
        output = 'MATH'
    elif 'electrical' in string:
        output = 'ECE'
    elif 'bio' in string:
        output = 'BIO'
    elif 'chem' in string:
        output = 'CHEM'
        
    # Otherwise, if uncaught - keep as is
    else:
        output = string
    
    return output


# In[12]:


df['major'] = df['major'].apply(standardize_major)


# In[13]:


df['major'].unique()


# Now, we want to standarize gender responses: 

# In[14]:


df['gender'].value_counts()


# In[15]:


def standardize_gender(gender):
    
    print(gender)
    string = gender
    # Make the input all lowercase
    string = string.lower()
    

    string = string.strip()
    
    if 'female' in string:
        output = 'female'
    elif 'f' in string:
        output = 'female'
    elif 'woman' in string:
        output = 'female'
    elif 'famale' in string:
        output = 'female'
    elif 'women' in string:
        output = 'female'
    elif 'male' in string:
        output = 'male'
    elif 'm' in string:
        output = 'male'
    elif 'man' in string:
        output = 'male'
    elif 'men' in string:
        output = 'male'
    elif 'nonbinary' in string:
        output = 'nonbinary_or_trans'
    elif 'transgender' in string:
        output = 'nonbinary_or_trans'
    # Otherwise, if uncaught - keep as is
    else:
        output = np.nan
    
    return output


# In[16]:


#df['gender']
df['gender'] = df['gender'].apply(standardize_gender)
df = df.dropna(subset = ['gender'])


# In[17]:


df['gender'].unique()


# In the data, there is a number of unique responses in the 'year' column:

# In[18]:


num_unique_responses = df['year'].nunique()
num_unique_responses


# In[19]:


df['year'].unique()


# Now, we can standarize the year column: 

# In[20]:


#START CODE 
def standardize_year(string):
    
    # Make the input all lowercase
    string = string.lower()
    
    # Drop all whitespace
    string = string.strip()
    
    #replace any occurences of 'first' with '1'
    string = string.replace('first', '1')

    #replace any occurences of 'second' with '2'
    string = string.replace('second', '2')    

    #replace any occurences of 'third' with '3'
    string = string.replace('third', '3')    

    #replace any occurences of 'fourth' with '4'
    string = string.replace('fourth', '4')    

    
    #replace any occurences of 'fifth' with '5'
    string = string.replace('fifth', '5')  

    #replace any occurences of 'sixth' with '6'
    string = string.replace('sixth', '6')    

    #replace any occurences of 'freshman' with '1'
    string = string.replace('freshman', '1')    

    #replace any occurences of 'sophomore' with '2'
    string = string.replace('sophomore', '2')    

    #replace any occurences of 'junior' with '3'
    string = string.replace('junior', '3')    

    #replace any occurences of 'senior' with 4'
    string = string.replace('senior', '4')    

    #replace any occurences of 'year' with '' (remove it from the string)
    string = string.replace('year', '')    

    #replace any occurences of 'th' with '' (remove it from the string)
    string = string.replace('th', '')    

    #replace any occurences of 'rd' with '' (remove it from the string)
    string = string.replace('rd', '')    

    #replace any occurences of 'nd' with '' (remove it from the string)
    string = string.replace('nd', '')    

    #strip the string of all leading and trailing whitespace (again)
    string = string.strip()

    #If the resulting string is a number and it is less than 10, then cast it into an integer and return that value
    # Cast to integer, if what's left seems appropriate
    if string.isnumeric() and int(string) < 10:
        out_put = int(string)
        
    #Else return np.nan to symbolize that the student's response was not a valid entry
    else:
        out_put = np.nan
    
    return out_put
#END CODE 


# In[21]:


df['year'].unique()


# In[22]:


df['year'] = df['year'].apply(standardize_year)


# What about weight? I can also standarize that column: 

# In[23]:


df['weight'] = df['weight'].astype(str)


# In[24]:


df['weight'].unique()


# In[25]:


def standardize_weight(string):
    
    #convert all characters of the string into lowercase
    string = string.lower()
    
    #strip the string of all leading and trailing whitespace
    string = string.strip()

    #replace any occurences of 'lbs' with '' (remove it from the string)
    string = string.replace('lbs', '')    

    #replace any occurences of 'lb' with '' (remove it from the string)
    string = string.replace('lb', '')    
    
    #replace any occurences of 'pounds' with '' (remove it from the string)
    string = string.replace('pounds', '')    

    if 'kg' in string:
        string = string.replace('kg', '')
        #strip the string of all leading and trailing whitespace
        string = string.strip()
        
        #can keep string 
        string = float(string)
        string = string * 2.2
    try: 
        return int(string)
    except:
        return np.nan


# In[26]:


df['weight'] = df['weight'].apply(standardize_weight)


# In[27]:


df['weight'].unique()


# Finall, the last column I want to standarize is height: 

# In[28]:


# convert all values to inches
def standardize_height(string):
    
    orig = string
    output = None
    
    # Basic string pre-processing
    string = string.lower()
    string = string.strip()
    
    string = string.replace('foot', 'ft')
    string = string.replace('feet', 'ft')
    string = string.replace('inches', 'in')
    string = string.replace('inch', 'in')
    string = string.replace('meters', 'm')
    string = string.replace('meter', 'm')
    string = string.replace('centimeters', 'cm')
    string = string.replace('centimeter', 'cm')
    string = string.replace(',', '')
    string = string.strip()
    
    # CASE 1: string is written in the format FEET <DIVIDER> INCHES
    dividers = ["'", "ft", "’", '”', '"','`', "-", "''"]
    
    for divider in dividers:
        
        # Split it into its elements
        elements = string.split(divider)

        # If the divider creates two elements
        if (len(elements) >= 2) and ((len(string) -1) != string.find(divider)):
            feet = elements[0]
            inch = elements[1] if elements[1] is not '' else '0'
            
            # Cleaning extranious symbols
            for symbol in dividers:
                feet = feet.replace(symbol, '')
                inch = inch.replace(symbol, '')
                inch = inch.replace('in','')
            
            # Removing whitespace
            feet = feet.strip()
            inch = inch.strip()
            
            # By this point, we expect 'feet' and 'inch' to be numeric
            # If not...we ignore this case
            if feet.replace('.', '').isnumeric() and inch.replace('.', '').isnumeric():
                
                # Converting feet to inches and adding it to the current inches
                output = (float(feet) * 12) + float(inch)
                break
            
    # CASE 2: string is written in the format FEET ft INCHES in 
    if ('ft' in string) and ('in' in string):
        
        # Split it into its elements
        elements = string.split('ft')
        feet = elements[0]
        inch = elements[1]
        
        # Removing extraneous symbols and stripping whitespace
        inch = inch.replace('inch', '')
        inch = inch.replace('in', '')
        feet = feet.strip()
        inch = inch.strip()
        
        # By this point, we expect 'feet' and 'inch' to be numeric
        # If not...we ignore this case
        if feet.replace('.', '').isnumeric() and inch.replace('.', '').isnumeric():
                
            # Converting feet to inches and adding it to the current inches
            output = (float(feet) * 12) + float(inch)
        
    # CASE 3: answer was given ONLY in cm
    #  Convert to inches: approximately 0.39 inches in a meter
    elif 'cm' in string:
        centimeters = string.replace('cm', '')
        centimeters = centimeters.strip()
        
        if centimeters.replace('.', '').isnumeric():
            output = float(centimeters) * 0.39
        
    # CASE 4: answer was given ONLY in meters
    #  Convert to inches: approximately 39 inches in a meter
    elif 'm' in string:
        
        meters = string.replace('m', '')
        meters = meters.strip()
        
        if meters.replace('.', '').isnumeric():
            output = float(meters)*39
        
    # CASE 5: answer was given ONLY in feet
    elif 'ft' in string:

        feet = string.replace('ft', '')
        feet = feet.strip()
        
        if feet.replace('.', '').isnumeric():
            output = float(feet)*12
    
    # CASE 6: answer was given ONLY in inches
    elif 'in' in string:
        inches = string.replace('in', '')
        inches = inches.strip()
        
        if inches.replace('.', '').isnumeric():
            output = float(inches)
        
    # CASE 7: answer not covered by existing scenarios / was invalid. 
    #  Return NaN
    if not output:
        output = np.nan

    return output


# In[29]:


# Applying the transformation and dropping invalid rows
df['height'] = df['height'].apply(standardize_height)
df = df.dropna(subset=['height'])


# In[30]:


df['height'].unique()


# In[31]:


df['age'] = df['age'].astype(np.int64)
df['age']


# # EDA

# In[32]:


fig = pd.plotting.scatter_matrix(df, alpha=0.2, figsize = (10,10))


# In[33]:


plt.rcParams['figure.figsize'] = (20,20)
ax = sns.countplot(x='major', data=df, color = 'brown')

#add title: 
ax.set_title('Number Students in Each Major')
#add y label 
ax.set_ylabel('Count')
#add x label 
ax.set_xlabel('How comfortable are you with statistics?');
ax.tick_params(axis='x', rotation=30)

f1 = plt.gcf()


# Below, I can also see a histogram of data responses from the dataframe: 

# In[34]:


plt.rcParams['figure.figsize'] = (16,10) #default plot size to output
sns.histplot(data = df, bins=30)
f2 = plt.gcf()


# What's the range of ages in the dataset? 

# In[35]:


r_age = df['age'].max() - df['age'].min()
print("The range of ages in the dataset are: ", r_age)


# Lets save the height data for both cognitive science and computer science majors:

# In[36]:


h_co = df[(df.major == 'COGSCI')]['height']
h_cs = df[(df.major == 'COMPSCI')]['height']


# In[37]:


h_co.describe()


# In[38]:


h_cs.describe()


# In[39]:


avg_h_co = h_co.agg('mean')
avg_h_co


# In[40]:


avg_h_cs = h_cs.agg('mean')
avg_h_cs


# In[41]:


print('Average height of cogs majors is \t {:2.2f} inches'.format(avg_h_co))
print('Average height of cs majors is \t\t {:2.2f} inches'.format(avg_h_cs))


# I want to use the `ttest_ind` function) to compare the two height distributions (`h_co` vs `h_cs`)
# 
# `ttest_ind` returns a t-statistic, and a p-value. Save these outputs to `t_val` and `p_val` respectively. 

# In[42]:


t_val, p_val = stats.ttest_ind(h_co, h_cs)


# In[43]:


if p_val < 0.01:
    print('Data Science accomplished, there is a significant difference!')
else:
    print('There is NOT a significant difference!')


# #Save the counts for each gender for 'COGSCI' amd 'OCMPSCI' majors: 

# In[44]:


#Save the counts for each gender for 'COGSCI' majors to a variable called g_co
g_co = df[(df.major == 'COGSCI')]['gender'].value_counts()
#Save the counts for each gender for 'COMPSCI' majors to a variable called g_cs
g_cs = df[(df.major == 'COMPSCI')]['gender'].value_counts()


# Ratio of women in Cognitive Science and Computer Science majors:

# In[45]:


r_co = g_co['female']
r_cs = g_cs['female']
print(r_co,r_cs )


# In[46]:


chisq, p_val_chi = stats.chisquare(np.array([g_co.values, g_cs.values]), axis=None)

if p_val_chi < 0.01:
    print('There is a significant difference in ratios!')


# Now I want to create a linear model which predicts height from majors. To do this I created a new dataframe containing only the majors of cognitive and computer science

# In[47]:


df2 = df[(df['major'] == 'COGSCI') | (df['major'] == 'COMPSCI')]


# In[48]:


df2


# In[49]:


#create the design matrices 
outcome_1, predictors_1 = patsy.dmatrices('height ~ major', df2)
#create the OLS model 
mod_1 = sm.OLS(outcome_1, predictors_1)
#fit the model 
res_1 = mod_1.fit()


# In[50]:


print(res_1.summary())

