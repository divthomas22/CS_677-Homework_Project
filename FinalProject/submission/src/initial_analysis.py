"""
Divya Thomas 
Class: CS 677
Date: 4/10/2023
Description of Problem (just a 1-2 line summary!): 
Read the song data into a dataframe and perform initial calculations and analysis of the data
in order to get the best features to use in predictions.
"""
import pandas as pd 
import math
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#function to convert the csv data into a pandas dataframe
def create_pandas_dataframe():

    #get file path
    input_dir = os.path.abspath(__file__) + '\..\datasets'
    file = os.path.join(input_dir, 'song_data.csv')

    #convert to df
    df = pd.read_csv(file)

    #This data contains duplicates, so in order to ensure the correct results and calculations, we need to clean it up
    df = df.drop_duplicates()

    #drop the song name column since it is not a continuous feature that will be tested
    df = df.drop('song_name', axis=1)

    return df


# calculate the mean and standard deviation of the popularity column 
# Function to compute the mean of a list of numbers
def get_mean(column):
    sum = 0
    for item in column:
        sum += item
    mean = sum / len(column)
    return mean

# Function to compute the standard deviation of a list of numbers
def get_sd(column):
    sum = 0
    mean = get_mean(column)
    for item in column:
        item = item**2
        sum += item
    sd = math.sqrt((sum / len(column)) - mean**2)
    return sd

# Use the mean as the limit for the high to low popularity
# In other words, anything below the mean is low popularity (-) and anything above is high (+) 
# Add a binary class variable to indicate popularity class of songs
def add_binary_class(df, limit):
    # Add a binary class variable to the set (Popularity > mean = '+', otherwise '-')
    df["rating"] = np.where(df["song_popularity"] > limit, '+', '-')
    return df 

# a function for other files to call to retrieve the dataset
def dataset():
    df = create_pandas_dataframe()
    df = add_binary_class(df, 50)
    return df

# split dataset 50/50 into training and testing data 
def split(df):
    x = df[['audio_valence', 'acousticness', 'energy', 'instrumentalness', 'loudness']]
    y = df['rating']

    x_train,x_test,y_train,y_test = train_test_split(
        x, y, 
        test_size=0.5)
    
    train = (x_train, y_train)
    test = (x_test, y_test)
    return (train, test)

def accuracy(dataframe):
    # get the list of actual NSP class values
    actual = dataframe['Actual'].tolist()
    # get the list of the predicted values 
    pred = dataframe["Prediction"].tolist()
    # get the total count of the predicted values 
    tot_count = len(pred)
    # set a counter for the correct predictions
    success_count = 0

    #match up each index in actual with predictions and increment success count if they match
    for i in range(0, tot_count):
        if (actual[i] == pred[i]):
            success_count += 1
    # return the percentage of success
    return (success_count / tot_count) * 100

def compute_confusion_matrix(df):
    # compute the confusion matrix
    cm = confusion_matrix(df['Actual'], df['Prediction'])
    return cm


# get TP, FP, TN, FN
def calc_label_accuracies(df):
    # get the list of true labels 
    true_list = df['Actual'].tolist()
    # get the list of the predicted labels
    pred_list = df["Prediction"].tolist()

    #set up counters for each scenario
    tp_count = 0
    fp_count = 0
    tn_count = 0
    fn_count = 0

    for i in range(len(pred_list)):
        if true_list[i] == '+':
            if pred_list[i] == '+': #true positive (true normal)
                tp_count += 1
            elif pred_list[i] == '-': #false negative
                fn_count += 1
        elif true_list[i]   == '-': 
            if pred_list[i] == '-': #true negative
                tn_count += 1
            elif pred_list[i] == '+': #false positive
                fp_count += 1
    count_list = [tp_count, fn_count, tn_count, fp_count]            
    return count_list

# function to calculate the True Positive Rate based off of the count list provided
def calc_tpr(count_list):
    # TPR = TP/(TP+FN)
    tp = count_list[0]
    fn = count_list[1]
    #based off of calc_label_accuracies function

    tpr = tp / (tp + fn)
    return tpr

# function to calculate the True Negative Rate based off of the count list provided
def calc_tnr(count_list):
    # TNR = TN/(TN+FP)
    tn = count_list[2]
    fp = count_list[3]
    #based off of calc_label_accuracies function

    tnr = tn / (tn + fp)
    return tnr

# compute the mean and standard deviation of each of the features tested
def compute_table(df):
    for column_name, column_data in df.items():
        mean = get_mean(column_data)
        sd = get_sd(column_data)
        print('%s - mean: %.2f, sd: %.2f' % (column_name, mean, sd) )
    print()


def get_default_probability(dataframe):
    # get the ratings data
    counts = dataframe["rating"].value_counts()
    # get count of '+' occurrences
    pos_count = counts.loc['+']
    # get count of '+' and '-' occurrences
    tot_count = dataframe["rating"].count()

    #calulate probability
    probability = (pos_count/tot_count)
    ret = 0
    if (probability >= 0.5):
        ret = '+'
    else: 
        ret = '-'
    return ret



df= dataset()
pop_list = df["song_popularity"]
mean = get_mean(pop_list)
sd = get_sd(pop_list)
print("Popularity mean:", mean, "\nPopularity standard deviation:", sd)
print(df)


plus_df = df[df['rating'] == '+']
minus_df = df[df['rating'] == '-']
compute_table(df[['audio_valence', 'acousticness', 'energy', 'instrumentalness', 'loudness']])
compute_table(plus_df[['audio_valence', 'acousticness', 'energy', 'instrumentalness', 'loudness']])
compute_table(minus_df[['audio_valence', 'acousticness', 'energy', 'instrumentalness', 'loudness']])
