"""
Divya Thomas 
Class: CS 677
Date: 4/1/2023
Homework Problem #1
Description of Problem (just a 1-2 line summary!): 
Create a dataframe with the banknote data and calculate the stats of each class type.
"""
import pandas as pd 
import numpy as np
import math
import os

# Question 1.1 - Read the banknote data into a pandas dataframe and add a "Color" column
print ("\n--Question 1.1--")
#function to convert the txt data into a pandas dataframe
def create_pandas_dataframe():
    #list of columns
    columns = ['f1', 'f2', 'f3', 'f4', 'class']
    #get file path
    input_dir = os.path.abspath(__file__) + '\..\datasets'
    bn_file = os.path.join(input_dir, 'data_banknote_authentication.txt')

    #convert to df using col names above
    df = pd.read_csv(bn_file, names=columns)

    # add color col
    df["color"] = np.where(df["class"]==0, 'green', 'red')

    return df

bn_df = create_pandas_dataframe()
print (bn_df)

# Question 1.2 - Compute the mean and sd of each column of class 0 or 1 subsets
print ("\n--Question 1.2--")

# Function to compute the mean of a list of numbers
def get_mean(values):
    sum = 0
    for item in values:
        sum += item
    mean = sum / len(values)
    return mean

# Function to compute the standard deviation of a list of numbers
def get_sd(values):
    sum = 0
    mean = get_mean(values)
    for item in values:
        item = item**2
        sum += item
    sd = math.sqrt((sum / len(values)) - mean**2)
    return sd

def calc_df_stats(df):
    # get data for all class subsets
    df_0 = df[df['class'] == 0]
    df_1 = df[df['class'] == 1]

    # list of dataframes 
    dataframes = [df, df_0, df_1]
    for i in range(len(dataframes)):
        #print which dataframe the following data will be from
        if i == 0:
            print ("\nAll Data:")
        elif i == 1:
            print ("\nClass 0 Data:")
        elif i == 2:
            print ("\nClass 1 Data:")
        curr_df = dataframes[i]
        for j in range(1, 5):
            datalist = curr_df['f'+str(j)].tolist()
            print('f%s mean: %.2f' % (j, get_mean(datalist)))
            print('f%s standard deviation: %.2f' % (j, get_sd(datalist)))

calc_df_stats(bn_df)

# Question 1.3 - please see supplemental documentation