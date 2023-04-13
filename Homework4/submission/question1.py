"""
Divya Thomas 
Class: CS 677
Date: 4/8/2023
Homework Problem #1
Description of Problem (just a 1-2 line summary!): 
Seperate the data into deceased and surviving patients and visualize their correlation factors.
"""
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Question 1.1 - Read the patient data into two seperate dataframes (surviving and deceased patients)
print ("\n--Question 1.1--")
#function to convert the csv data into a pandas dataframe
def create_pandas_dataframes():

    #get file path
    input_dir = os.path.abspath(__file__) + '\..\datasets'
    file = os.path.join(input_dir, 'heart_failure_clinical_records_dataset.csv')

    #convert to df
    df = pd.read_csv(file)

    df_0 = df[df['DEATH_EVENT'] == 0] #surviving patients 
    df_1 = df[df['DEATH_EVENT'] == 1] #deceased patients

    return (df_0, df_1)

df_0, df_1 = create_pandas_dataframes()

print('Surviving patients: \n', df_0)
print('\nDeceased patients: \n', df_1)

# Question 1.2 - Create a correlation matrix and visual representation of the dataset
print ("\n--Question 1.2--")
def create_corr_plot(df, filename):
    #get file path to save plot to
    input_dir = os.path.abspath(__file__) + '\..\datasets'
    file = os.path.join(input_dir, filename)

    #construct the correlation matrix
    m = df.corr()

    # Visualize correlation matrix
    plt.figure(figsize=(10,10))
    plot = sns.heatmap(df.corr().round(3), annot = True)
 
    #save to file
    plt.savefig(file)

    print("Saving file %s to datasets directory..." % filename)

    return m

m0 = create_corr_plot(df_0, "surviving_patients.png")
m1 = create_corr_plot(df_1, "deceased_patients.png")

print(m0, '\n', m1)


# Question 1.3 - See supplemental documentation.