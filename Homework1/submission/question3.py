"""
Divya Thomas 
Class: CS 677
Date: 3/18/2023
Homework Problem #3
Description of Problem (just a 1-2 line summary!):
Compute and analyze the aggregate table for all 5 years for TGT and SPY
"""

import question1 as q1
import read_stock_data_from_file as rdf

# Get the dataset for 'TGT' and 'SPY'
tgt_file =rdf.create_dataset('TGT')
spy_file =rdf.create_dataset('SPY')

# List of weekdays and years to get return results for
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
years = ['2018', '2019', '2020', '2021', '2022']

#Get all returns for a dataset - all years
print("\nQuestion 3")
def get_all_agg_returns(dataset, weekday):
    return_list = []
    for year in years:
        #get all of the returns for all of the years
        return_list.extend(q1.get_all_returns(dataset, weekday, year))
    return return_list

#Same with negative returns and non-negative returns
def get_agg_neg_returns(dataset, weekday):
    return_list = []
    for year in years:
        #get all of the returns for all of the years
        return_list.extend(q1.get_neg_returns(dataset, weekday, year))
    return return_list

def get_agg_nonneg_returns(dataset, weekday):
    return_list = []
    for year in years:
        #get all of the returns for all of the years
        return_list.extend(q1.get_nonneg_returns(dataset, weekday, year))
    return return_list


# Function to output all the requested info for a specific day of the week of all years
def output_all_info(dataset, day):
    r = get_all_agg_returns(dataset, day)
    r_neg = get_agg_neg_returns(dataset, day)
    r_pos = get_agg_nonneg_returns(dataset, day)
    print('-- Statistics for ' + day +' --')
    print('|R|: ', str(len(r)))
    print('R Mean: ', str(q1.get_mean(r)))
    print('R Standard Deviation: ', str(q1.get_sd(r)))
    print('|R-|: ', str(len(r_neg)))
    print('R- Mean: ', str(q1.get_mean(r_neg)))
    print('R- Standard Deviation: ', str(q1.get_sd(r_neg)))
    print('|R+|: ', str(len(r_pos)))
    print('R+ Mean: ', str(q1.get_mean(r_pos)))
    print('R+ Standard Deviation: ', str(q1.get_sd(r_pos)) + '\n')

print('\nTGT Aggregated Data: ')
for weekday in weekdays:
    output_all_info(tgt_file, weekday)


print('\nSPY Aggregated Data: ')
for weekday in weekdays:
    output_all_info(spy_file, weekday)

#Question 3.1 - Compute the best and worst days for each aggregated set
print("\nQuestion 3.1")

#Similar to the functions in question 2, but without year dependency
def get_best_weekday_agg(dataset):
    best_mean = -1
    best_weekday = ''
    for weekday in weekdays:
        mean = q1.get_mean(get_all_agg_returns(dataset, weekday))
        if mean > best_mean:
            best_mean = mean
            best_weekday = weekday
    print("Best Weekday: " + best_weekday )

def get_worst_weekday_agg(dataset):
    worst_mean = 1
    worst_weekday = ''
    for weekday in weekdays:
        mean = q1.get_mean(get_all_agg_returns(dataset, weekday))
        if mean < worst_mean:
            worst_mean = mean
            worst_weekday = weekday
    print("Worst Weekday: " + worst_weekday )

print('\nTGT Best/Worst: ')
get_best_weekday_agg(tgt_file)
get_worst_weekday_agg(tgt_file)

print('\nSPY Best/Worst: ')
get_best_weekday_agg(spy_file)
get_worst_weekday_agg(spy_file)

# Question 3.2 : Answered on summary word document