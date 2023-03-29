"""
Divya Thomas 
Class: CS 677
Date: 3/24/2023
Homework Problem #2
Description of Problem (just a 1-2 line summary!):
Use the data from years 1-3 to predict data in years 4-5 regarding the stock returns.
"""

import question1 as q1


tgt_df = q1.create_pandas_dataframe('TGT')
q1.add_true_label(tgt_df)
spy_df = q1.create_pandas_dataframe('SPY')
q1.add_true_label(spy_df)

print ("\n--Question 2.1--")
# Question 2.1 - Predict the next day True Label for all Year 4-5 days using Year 1-3 data based on W
def predict_w(dataframe, w):
    # get test data 
    df_test = dataframe[dataframe['Year'] > 2020]
    #convert the true label column to a list of "-" and "+" values
    label_list = df_test["True Label"].tolist()
    #create a list to hold the data for the W column
    w_list = []
    #iterate through the test data
    for i in range(len(label_list)):
        prediction = float('nan')
        #if we have to go into training data to get prev day vals, NaN
        #otherwise, we can move forward
        if (i-w >= 0):
            # get the base sequence to predict the next day from
            seq = label_list[i-w: i]
            prediction = check_prob(dataframe, seq)
        w_list.append(prediction)
    return w_list

#function to return the most probable next label based off training data
def check_prob(dataframe, seq):
    # get training data 
    df_3yr = dataframe[dataframe['Year'] < 2021]
    #convert the true label column to a list of "-" and "+" values
    label_list = df_3yr["True Label"].tolist()
    #set each type of sequence to check for from the base seq param
    seq_plus = seq + ['+']
    seq_minus = seq + ['-']

    #set counters for each type of seq
    plus_days = 0
    minus_days = 0 

    #iterate through all training data 
    for i in range(len(seq), len(label_list)):
        # count occurences of each seq type in the data
        if (label_list[i-len(seq): i+1] == seq_plus):
            plus_days += 1
        elif (label_list[i-len(seq): i+1] == seq_minus):
            minus_days += 1
    
    prediction = float('nan')
    # if occurrences of each type is equal, use default probability
    if plus_days == minus_days:
        default = q1.compute_default_probability(dataframe)
        #if this is higher than .50, and up day is predicted
        if (default > 0.5):
            prediction = '+'
        else:
            prediction = '-'
    # otherwise, go with which ever has the highest count
    elif plus_days > minus_days:
        prediction = '+'
    else:
        prediction = '-'
    return prediction

#function to call predict_w for all w from 2-4 adn consolidate results
def predict_w2_4(dataframe):
    # get test data 
    df_test = dataframe[dataframe['Year'] > 2020].copy()
    for i in range(2,5):
        df_test["W" + str(i)] = predict_w(dataframe, i)
    return df_test[['Date', 'True Label', 'W2', 'W3', 'W4']]


tgt_w2_4 = predict_w2_4(tgt_df)
spy_w2_4 = predict_w2_4(spy_df)

print("\nTGT W2-W4 Data:\n", tgt_w2_4)
print("\nSPY W2-W4 Data:\n", spy_w2_4)

print ("\n--Question 2.2--")
# Question 2.2 - Calculate the accuracy of the predictions made above 
def w_accuracy(dataframe, w):
    # get the list of true labels 
    true_list = dataframe['True Label'].tolist()
    # get the list of the predicted labels for the w param passed
    label_list = dataframe["W" + str(w)].tolist()
    # get the total count of the predicted labels (not including NaNs)
    tot_count = len(label_list)
    start = 0
    if type(w) == int():
        tot_count = len(label_list) - w
        start = w
    # set a counter for the correct predictions
    success_count = 0

    #match up each index in true_list with label_list and increment success count if they match
    for i in range(start, tot_count):
        if (true_list[i] == label_list[i]):
            success_count += 1
    # return the percentage of success
    return (success_count / tot_count) * 100
    

print("\nTGT: ")
print("W2 computations were %.2f%% accurate" % w_accuracy(tgt_w2_4, 2))
print("W3 computations were %.2f%% accurate" % w_accuracy(tgt_w2_4, 3))
print("W4 computations were %.2f%% accurate" % w_accuracy(tgt_w2_4, 4))

print("\nSPY: ")
print("W2 computations were %.2f%% accurate" % w_accuracy(spy_w2_4, 2))
print("W3 computations were %.2f%% accurate" % w_accuracy(spy_w2_4, 3))
print("W4 computations were %.2f%% accurate" % w_accuracy(spy_w2_4, 4))

# Question 2.3 : Answered on summary word document