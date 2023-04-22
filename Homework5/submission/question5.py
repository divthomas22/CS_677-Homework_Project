"""
Divya Thomas 
Class: CS 677
Date: 4/15/2023
Homework Problem #5
Description of Problem (just a 1-2 line summary!):
Calculate TP. TN, FP, FN, TPR, and TNR for all classifier tested in HW5 (see outputs on other files).
This file contains only the functions used.
"""

# Question 5 

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
        if true_list[i] == 1:
            if pred_list[i] == 1: #true positive (true normal)
                tp_count += 1
            elif pred_list[i] == 0: #false negative
                fn_count += 1
        elif true_list[i]   == 0: 
            if pred_list[i] == 0: #true negative
                tn_count += 1
            elif pred_list[i] == 1: #false positive
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

