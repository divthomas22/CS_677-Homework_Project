"""
Divya Thomas 
Class: CS 677
Date: 3/18/2023
Homework Problem #5
Description of Problem (just a 1-2 line summary!):
You use a "Buy-and-Hold" strategy instead of listening to your oracle.
Create a function that calculates the amount you end up with after 5 years.
"""

import read_stock_data_from_file as rdf

# Get the dataset for 'TGT' and 'SPY'
tgt_file =rdf.create_dataset('TGT')
spy_file =rdf.create_dataset('SPY')

print("\nQuestion 5.1")
# Question 5.1 - Similar to question 4, but will include the effects of negative returns
def buy_and_hold(dataset, amount):
    #iterate through all the days
    for i in range(len(dataset)-1):
        next_day = float(dataset[i+1]["Return"]) # get the next day's return
        # add the return percentage to your amount
        amount += amount*next_day
    #once done, output the final amount
    print("You will have $%.2f on the last trading day of 2022" % amount)
        
print ("TGT - ", end="")
buy_and_hold(tgt_file, 100)

print ("SPY - ", end="")
buy_and_hold(spy_file, 100)

# Question 5.2 : Answered on summary word document