"""
Divya Thomas 
Class: CS 677
Date: 3/24/2023
Homework Problem #3
Description of Problem (just a 1-2 line summary!):
Try to improve your predictions for years 4-5 by using ensemble learning.
"""
import question1 as q1
import question2 as q2
import statistics as s


tgt_df = q1.create_pandas_dataframe('TGT')
q1.add_true_label(tgt_df)
spy_df = q1.create_pandas_dataframe('SPY')
q1.add_true_label(spy_df)

tgt_w2_4 = q2.predict_w2_4(tgt_df)
spy_w2_4 = q2.predict_w2_4(spy_df)

print ("\n--Question 3.1--")
# compute the ensemble label of the dataset using W* values 
def ensemble(df):
    # set up a list to store WE (W-ensemble) labels to
    we_list = []
    # iterate through the dataframe
    for index,row in df.iterrows():
        label_list = [] # create a list of all the labels
        label_list.append(row["W2"])
        label_list.append(row["W3"])
        label_list.append(row["W4"])
        mode = s.mode(label_list) # get the mode in the list
        we_list.append(mode) # add the mode to the we list
    #now add the we_list to the dataframe
    df["WE"] = we_list

    return df

tgt_we = ensemble(tgt_w2_4)
spy_we = ensemble(spy_w2_4)

print("Ensemble values for TGT: \n", tgt_we)
print("Ensemble values for SPY: \n", spy_we)

# Question 3.2 - Calculate the accuracy of the predictions made above 
print ("\n--Question 3.2--")
print("TGT: ", end='')
print("WE computations were %.2f%% accurate" % q2.w_accuracy(tgt_we, 'E'))
print("SPY: ", end='')
print("WE computations were %.2f%% accurate" % q2.w_accuracy(spy_we, 'E'))

# Question 3.3-4 - Calculate the accuracy of each label
print ("\n--Question 3.3 & 3.4--")

def label_w_accuracy(dataframe, w):
    # get the list of true labels 
    true_list = dataframe['True Label'].tolist()
    # get the list of the predicted labels for the w param passed
    label_list = dataframe["W" + str(w)].tolist()
    # get the total count of the predicted labels 
    tot_up_count = 0
    tot_down_count = 0
    start = 0
    # set a counter for the correct predictions
    success_up_count = 0
    success_down_count = 0

    #match up each index in true_list with label_list and increment success count if they match
    for i in range(start, len(label_list)):
        if (true_list[i] == '+'):
            tot_up_count += 1
            if (label_list[i] == '+'):
                success_up_count += 1
        elif (true_list[i] == '-'):
            tot_down_count  += 1
            if (label_list[i] == '-'):
                success_down_count += 1
    # return the percentage of success
    percent_up = (success_up_count / tot_up_count) * 100
    percent_down = (success_down_count / tot_down_count) * 100
    return (percent_up, percent_down)

tgt_a_w2 = label_w_accuracy(tgt_we, 2)
tgt_a_w3 = label_w_accuracy(tgt_we, 3)
tgt_a_w4 = label_w_accuracy(tgt_we, 4)
tgt_a_we = label_w_accuracy(tgt_we, 'E')

spy_a_w2 = label_w_accuracy(spy_we, 2)
spy_a_w3 = label_w_accuracy(spy_we, 3)
spy_a_w4 = label_w_accuracy(spy_we, 4)
spy_a_we = label_w_accuracy(spy_we, 'E')


print("TGT:")
print("W2 computations predicted %.2f%% of up days and %.2f%% of down days" % (tgt_a_w2[0], tgt_a_w2[1]))
print("W3 computations predicted %.2f%% of up days and %.2f%% of down days" % (tgt_a_w3[0], tgt_a_w3[1]))
print("W4 computations predicted %.2f%% of up days and %.2f%% of down days" % (tgt_a_w4[0], tgt_a_w4[1]))
print("WE computations predicted %.2f%% of up days and %.2f%% of down days" % (tgt_a_we[0], tgt_a_we[1]))

print("\nSPY:")
print("W2 computations predicted %.2f%% of up days and %.2f%% of down days" % (spy_a_w2[0], spy_a_w2[1]))
print("W3 computations predicted %.2f%% of up days and %.2f%% of down days" % (spy_a_w3[0], spy_a_w3[1]))
print("W4 computations predicted %.2f%% of up days and %.2f%% of down days" % (spy_a_w4[0], spy_a_w4[1]))
print("WE computations predicted %.2f%% of up days and %.2f%% of down days" % (spy_a_we[0], spy_a_we[1]))

