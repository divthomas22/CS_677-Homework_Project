"""
Divya Thomas 
Class: CS 677
Date: 3/24/2023
Homework Problem #5
Description of Problem (just a 1-2 line summary!):
Using different prediction strategies plot and examine the growth of 
your stock across the 2 test years when starting with $100
"""

import question1 as q1
import question2 as q2
import question3 as q3
import matplotlib.pyplot as plt

tgt_df = q1.create_pandas_dataframe('TGT')
q1.add_true_label(tgt_df)

# Question 5.1 - Plot the growth of your investment when you follow the predictions
# of your best W* and ensemble. Also plot for the buy-and-hold strategy.

#get list of end of day amounts for a w* or ensemble prediction
def w_ensemble(df, amount, w):
    amount_list = [amount]
    df_w2_4 = q2.predict_w2_4(df)
    df_we = q3.ensemble(df_w2_4)
    w_list = df_we["W" + str(w)].tolist() # w* label list
    date_list = df_we["Date"].tolist() # list for test dates
    df_test = df[df['Year'] > 2020]
    ret_list = df_test['Return'].tolist() # list of returns for dates
    #iterate through all the days 
    for i in range(len(date_list)):
        if (i < len(date_list) -1):
            # if the w* value predicts an up day tomorrow
            if (w_list[i+1] == '+'):
                #realize the next day's return 
                amount += amount*ret_list[i+1]
        if (i > 0):
            amount_list.append(amount)
            #otherwise do nothing
    return amount_list


# get list of end of day amounts for buy and hold
def buy_and_hold(full_df, amount):
    df = full_df[full_df['Year'] > 2020]
    date_list = df["Date"].tolist()
    amount_list = [amount]
    returns = df["Return"].tolist()
    #iterate through all the days
    for i in range(len(date_list)):
        if (i < len(date_list)-1):
            amount += amount*returns[i+1]
            amount_list.append(amount)
    return amount_list

#plot data for all values
def plot_data(x, bh, w, e):
    plt.plot(x, bh, label='Buy and Hold') #plot the buy and hold growth
    plt.plot(x, w, label='W3') # plot the w3 growth (which is the most accurate)
    plt.plot(x, e, label='Ensemble') # plot the ensemble growth

    plt.legend() #add plot legend

    plt.show() #show the plot
    

df_test = tgt_df[tgt_df['Year'] > 2020]
date_list = df_test["Date"].tolist() # list for test dates (x values)
bh_growth = buy_and_hold(tgt_df, 100) #buy and hold growth
w3 = w_ensemble(tgt_df, 100, 3) # w3 growth - since w3 has highest accuracy
we = w_ensemble(tgt_df, 100,'E') # ensemble growth 

plot_data(date_list, bh_growth, w3, we)


# Question 5.1-2 : See the plotin datasets\Q5.1_plot.png and on word document with 5.2 answer




