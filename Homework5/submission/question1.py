"""
Divya Thomas 
Class: CS 677
Date: 4/15/2023
Homework Problem #1
Description of Problem (just a 1-2 line summary!): 
Upload the raw data for the diagnostic features for CTGs performed and 
insert into a dataframe to be used for analysis on the relationship 
of normal cases and abnormal cases with the features: MSTV, Width, Mode, Variance
"""
import pandas as pd 
import numpy as np
import os

# Question 1.1 - Read the raw data into a dataframe and clean up any incomplete rows
print ("\n--Question 1.1--")
#function to convert the excel data into a pandas dataframe
def create_pandas_dataframe():
    #get file path
    input_dir = os.path.abspath(__file__) + '\..\datasets'
    file = os.path.join(input_dir, 'CTG.xls')
    #convert to df
    full_df = pd.read_excel(file, sheet_name='Raw Data')

    #remove any rows that contain any NaN values
    full_df = full_df.dropna()

    # add nsf_class column
    full_df = add_nsp_class(full_df)

    # Use only the columns that your group is assigned to analyze: 
    #  Group 3: MSTV, Width, Mode, Variance
    df = full_df[['MSTV', 'Width', 'Mode', 'Variance', 'NSP', 'NSP_CLASS']] 

    return df

# Question 1.2 - Add a binary classifier by organizing the NSP values:
# 1 - Normal(NSP = 1), 0 - Abnormal (NSP = anything else)
def add_nsp_class(df):
    df['NSP_CLASS'] = np.where(df['NSP'] == 1, 1, 0)
    return df

df = create_pandas_dataframe()
print(df)