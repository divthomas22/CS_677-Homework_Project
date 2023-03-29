"""
Divya Thomas 
Class: CS 677
Date: 3/24/2023
Homework Problem #1
Description of Problem (just a 1-2 line summary!): 
Using pandas and numpy, create a dataframe to calculate probabilities for the specified
scenarios below.
"""
import pandas as pd 
import numpy as np
import os

# Question 1.1 - Read the stock data into a Pandas DataFrame and add a "True Label" column
print ("\n--Question 1.1--")
#function to convert the cvs data into a pandas dataframe
def create_pandas_dataframe(ticker):
    input_dir = os.path.abspath(__file__) + '\..\datasets'
    ticker_file = os.path.join(input_dir, ticker + '.csv')
    df = pd.read_csv(ticker_file)
    print('Printing %s dataframe after file read...\n' % ticker, df)
    return df

#function to add 'True Label' column to the dataframe
def add_true_label(dataframe):
    dataframe["True Label"] = np.where(dataframe["Return"]<0, '-', '+')
    print('Adding True Label column to dataframe...\n', dataframe)

tgt_df = create_pandas_dataframe('TGT')
add_true_label(tgt_df)
spy_df = create_pandas_dataframe('SPY')
add_true_label(spy_df)


# Question 1.2 - Compute the default probability from the trading data of the first 3 years (2018-2020)
# that the next day is an "up" day
print ("\n--Question 1.2--")
def compute_default_probability(dataframe):
    # get the set of data only within 2018 - 2020
    df_3yr = dataframe[dataframe['Year'] < 2021]
    counts = df_3yr["True Label"].value_counts()
    # get count of '+' occurrences
    pos_count = counts.loc['+']
    # get count of '+' and '-' occurrences
    tot_count = df_3yr["True Label"].count()

    #calulate probability
    probability = (pos_count/tot_count)

    print("The default probability that the next day after 2020 is an 'up' day is", probability)

print("TGT: ", end='')
compute_default_probability(tgt_df)
print("SPY: ", end='')
compute_default_probability(spy_df)

# Question 1.3 - Calulcate the probability of a day being + for k = 1, 2, 3, where k is the number of 
# times "-" occurs before the day in question
# this question is also only referring to the first three years (2018-2020)
print ("\n--Question 1.3--")

def calc_k_neg_prob(dataframe, k):
    # get 3 yr data 
    df_3yr = dataframe[dataframe['Year'] < 2021]
    count_plus = 0 # num of "(-*k) +" occurrences
    count_minus = 0 # num of "(-*k) -" occurrences
    #convert the true label column to a list of "-" and "+" values
    label_list = df_3yr["True Label"].tolist()
    #iterate through the dataframe starting at k and go until the last date
    for i in range(k, len(label_list)):
        # check if the value at i index is "+" and everything between that and i-k is "-"
        if (label_list[i-k:i] == ["-"]*k) & (label_list[i] == "+"):
            count_plus += 1  # increment count_plus 

        # check if the value at i index is "-" and everything between that and i-k is "-"
        elif (label_list[i-k:i] == ["-"]*k) & (label_list[i] == "-"):
            count_minus += 1 #increment count_minus
    #calculate the probability 
    #( occurrences of (k'-')'+' / (occurrences of (k'-')'+' plus occurrences of (k'-')'-')
    probability = count_plus / (count_plus + count_minus)
    print ( "The probability of there being an 'up' day after %d consecutive 'down' day(s) is %.3f" % (k, probability) )

print("TGT: ")
calc_k_neg_prob(tgt_df, 1)
calc_k_neg_prob(tgt_df, 2)
calc_k_neg_prob(tgt_df, 3)

print("\nSPY: ")
calc_k_neg_prob(spy_df, 1)
calc_k_neg_prob(spy_df, 2)
calc_k_neg_prob(spy_df, 3)


# Question 1.4 - Calulcate the probability of a day being + for k = 1, 2, 3, where k is the number of 
# times "+" occurs before the day in question
# this question is also only referring to the first three years (2018-2020)
print ("\n--Question 1.4--")

def calc_k_pos_prob(dataframe, k):
    # get 3 yr data 
    df_3yr = dataframe[dataframe['Year'] < 2021]
    count_plus = 0 # num of "(+*k) +" occurrences
    count_minus = 0 # num of "(+*k) -" occurrences
    #convert the true label column to a list of "-" and "+" values
    label_list = df_3yr["True Label"].tolist()
    #iterate through the dataframe starting at k and go until the last date
    for i in range(k, len(label_list)):
        # check if the value at i index is "+" and everything between that and i-k is "+"
        if (label_list[i-k:i] == ["+"]*k) & (label_list[i] == "+"):
            count_plus += 1  # increment count_plus 

        # check if the value at i index is "-" and everything between that and i-k is "+"
        elif (label_list[i-k:i] == ["+"]*k) & (label_list[i] == "-"):
            count_minus += 1 #increment count_minus
    #calculate the probability 
    #( occurrences of (k'+')'+' / (occurrences of (k'+')'+' plus occurrences of (k'+')'-')
    probability = count_plus / (count_plus + count_minus)
    print ( "The probability of there being an 'up' day after %d consecutive 'up' day(s) is %.3f" % (k, probability) )

print("TGT: ")
calc_k_pos_prob(tgt_df, 1)
calc_k_pos_prob(tgt_df, 2)
calc_k_pos_prob(tgt_df, 3)

print("\nSPY: ")
calc_k_pos_prob(spy_df, 1)
calc_k_pos_prob(spy_df, 2)
calc_k_pos_prob(spy_df, 3)