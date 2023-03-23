"""
Divya Thomas 
Class: CS 677
Date: 3/18/2023
Homework Problem #6
Description of Problem (just a 1-2 line summary!):
The oracle from previous problems decided to sabotage you in different ways. 
Create functions to predict your final amount after each method of sabotage.
"""

import read_stock_data_from_file as rdf

# Get the dataset for 'TGT' and 'SPY'
tgt_file =rdf.create_dataset('TGT')
spy_file =rdf.create_dataset('SPY')

print("\nQuestion 6.1")
#Question 6.1 - scenario a - the 10 days with the best return values are ignored
def scenario_a(dataset, amount):
    # get the list of dates with the top 10 return values
    top_10 = best_days(dataset, 10)
    #iterate through all the days
    for i in range(len(dataset)-1):
        next_day = float(dataset[i+1]["Return"]) # get the next day's return
        # FIRST check if the next day is one of the top 10
        if dataset[i+1]["Date"] in top_10:
            # do nothing
            continue
        #otherwise, advise as usual
        elif next_day > 0: #if the return is positive
            # add the return percentage to your amount
            amount += amount*next_day
        #if any other return value, don't do anything
    #once done, output the final amount
    print("Scenario A: You will have $%.2f on the last trading day of 2022" % amount)



# A function to return a list of the 'n' number of days with the best return values
def best_days(dataset, n):
    # create a dict that will contain each date as a key and its return as a value
    dict = {}
    #iterate through the dataset
    for record in dataset:
        # get the date value to store as the key
        date = record["Date"]
        #get the return to store as the value
        ret = float(record["Return"])
        #add to the dictionary
        dict[date] = ret
    # now sort the dictionary by the return values, the key is the sorted value
    # this should return a list of tuples containing the date and return in descending order
    sort_dict = sorted(dict.items(), key=lambda x: x[1], reverse=True)

    #create a list to hold the top ten dates
    top_days = []
    # get the first n items in this list
    for i in range(n):
        #store the first tuple value (date) in the list
        top_days.append(sort_dict[i][0])
    #return the list
    return top_days



#Question 6.1 - scenario b - the 10 days with the worst return values are realized
def scenario_b(dataset, amount):
    # get the list of dates with the top 10 return values
    worst_10 = worst_days(dataset, 10)
    #iterate through all the days
    for i in range(len(dataset)-1):
        next_day = float(dataset[i+1]["Return"]) # get the next day's return
        # FIRST check if the next day is one of the worst 10 or if the return is positive
        if (dataset[i+1]["Date"] in worst_10) | (next_day > 0):
            # add the return percentage to your amount
            amount += amount*next_day
        #if any other return value, don't do anything
    #once done, output the final amount
    print("Scenario B: You will have $%.2f on the last trading day of 2022" % amount)



# A function to return a list of the 'n' number of days with the worst return values
def worst_days(dataset, n):
    # create a dict that will contain each date as a key and its return as a value
    dict = {}
    #iterate through the dataset
    for record in dataset:
        # get the date value to store as the key
        date = record["Date"]
        #get the return to store as the value
        ret = float(record["Return"])
        #add to the dictionary
        dict[date] = ret
    # now sort the dictionary by the return values, the key is the sorted value
    # this should return a list of tuples containing the date and return in ascending order
    sort_dict = sorted(dict.items(), key=lambda x: x[1], reverse=False)

    #create a list to hold the top ten dates
    bottom_days = []
    # get the first n items in this list
    for i in range(n):
        #store the first tuple value (date) in the list
        bottom_days.append(sort_dict[i][0])
    #return the list
    return bottom_days



#Question 6.1 - scenario c - 5 best days are ignored and 5 worst days are realized
def scenario_c(dataset, amount):
    # get the list of dates with the top 5 return values 
    top_5 = best_days(dataset, 5)
    # get the list of dates with the bottom 5 return values 
    bottom_5 = worst_days(dataset, 5)
    #iterate through all the days
    for i in range(len(dataset)-1):
        next_day = float(dataset[i+1]["Return"]) # get the next day's return
        # FIRST check if the next day is one of the top 5, do nothing on these
        if dataset[i+1]["Date"] in top_5:
            continue
        # if the next day is one of the bottom 5 dates or just a normal positive day
        elif (dataset[i+1]["Date"] in bottom_5) | (next_day > 0):
            # add the return percentage to your amount
            amount += amount*next_day
        #if any other return value, don't do anything
    #once done, output the final amount
    print("Scenario C: You will have $%.2f on the last trading day of 2022" % amount)



print ("TGT - ", end="")
scenario_a(tgt_file, 100)

print ("SPY - ", end="")
scenario_a(spy_file, 100)

print("\n")

print ("TGT - ", end="")
scenario_b(tgt_file, 100)

print ("SPY - ", end="")
scenario_b(spy_file, 100)

print("\n")

print ("TGT - ", end="")
scenario_c(tgt_file, 100)

print ("SPY - ", end="")
scenario_c(spy_file, 100)

# Question 6.2-3 : Answered on summary word document