"""
Divya Thomas 
Class: CS 677
Date: 3/18/2023
Homework Problem #4
Description of Problem (just a 1-2 line summary!): 
Create an oracle to check the next day's return to "predict" that the return gets
realized when the return is positive, and ignore when the return is negative.
"""

import read_stock_data_from_file as rdf

# Get the dataset for 'TGT' and 'SPY'
tgt_file =rdf.create_dataset('TGT')
spy_file =rdf.create_dataset('SPY')

print("Question 4.1 & 4.2")
# Question 4 Setup - an oracle provides the correct predictions and you follow its advice.
def successful_oracle(dataset, amount):
    #iterate through all the days
    for i in range(len(dataset)-1):
        next_day = float(dataset[i+1]["Return"]) # get the next day's return
        if next_day > 0: #if the return is positive
            # add the return percentage to your amount
            amount += amount*next_day
        #if any other return value, don't do anything
    #once done, output the final amount
    print("You will have $%.2f on the last trading day of 2022" % amount)
        
print ("TGT - ", end="")
successful_oracle(tgt_file, 100)

print ("SPY - ", end="")
successful_oracle(spy_file, 100)

