"""
Divya Thomas 
Class: CS 677
Date: 3/18/2023
Homework Problem #1
Description of Problem (just a 1-2 line summary!):
For each year in the 5-year stock statistics for TGT, compute the mean 
and standard deviation for all returns, all negative returns, and all 
non-negative returns for each day of the week and analyze the results.
"""

import read_stock_data_from_file as rdf

# Question 1.1 
print("Question 1.1")
# Function to compute the mean of a list of numbers
def get_mean(returns):
    sum = 0
    for item in returns:
        sum += item
    mean = sum / len(returns)
    return mean

# Function to compute the standard deviation of a list of numbers
def get_sd(returns):
    sum = 0
    mean = get_mean(returns)
    for item in returns:
        item = item**2
        sum += item
    sd = (sum / len(returns)) - mean**2
    return sd

# Function to get all returns for a specific day of the week of a year
def get_all_returns(dataset, day, year):
    return_list = []
    for record in dataset:
        # check that you only pull records with the correct year and weekday
        if (record['Year'] == year) & (record['Weekday'] == day):
            return_val = float(record['Return'])
            return_list.append(return_val)
    return return_list

# Function to get all negative returns for a specific day of the week of a year
def get_neg_returns(dataset, day, year):
    return_list = []
    for record in dataset:
        # check that you only pull records with the correct year, weekday, and negative return
        if (record['Year'] == year) & (record['Weekday'] == day) &(float(record['Return']) < 0):
            return_val = float(record['Return'])
            return_list.append(return_val)
    return return_list

# Function to get all non-negative returns for a specific day of the week of a year
def get_nonneg_returns(dataset, day, year):
    return_list = []
    for record in dataset:
        # check that you only pull records with the correct year, weekday, and non-negative return
        if (record['Year'] == year) & (record['Weekday'] == day) & (float(record['Return']) >= 0):
            return_val = float(record['Return'])
            return_list.append(return_val)
    return return_list


# Function to output all the requested info for a specific day of the week of a given year
def output_all_info(dataset, day, year):
    r = get_all_returns(dataset, day, year)
    r_neg = get_neg_returns(dataset, day, year)
    r_pos = get_nonneg_returns(dataset, day, year)
    print('-- Statistics for ' + day + ' in ' + year + ' --')
    print('|R|: ', str(len(r)))
    print('R Mean: ', str(get_mean(r)))
    print('R Standard Deviation: ', str(get_sd(r)))
    print('|R-|: ', str(len(r_neg)))
    print('R- Mean: ', str(get_mean(r_neg)))
    print('R- Standard Deviation: ', str(get_sd(r_neg)))
    print('|R+|: ', str(len(r_pos)))
    print('R+ Mean: ', str(get_mean(r_pos)))
    print('R+ Standard Deviation: ', str(get_sd(r_pos)))

# Get the dataset for 'TGT'
tgt_file =rdf.create_dataset('TGT')

# List of weekdays and years to get return results for
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
years = ['2018', '2019', '2020', '2021', '2022']

#get the return value stats for each weekday-year combination

for year in years:
    print ('-- %s --\n' % year)
    for weekday in weekdays:
        output_all_info(tgt_file, weekday, year)
        print('\n')
    print('\n')



# Question 1.3 - Determine if there are more non-negative vs negative returns 
print("Question 1.3")
# Across all 5 years
def neg_or_nonneg(dataset):
    print('Comparing return values across entire dataset...')
    neg_count = 0 
    nneg_count = 0
    for record in dataset:
        if float(record['Return']) < 0:
            neg_count += 1
        else:
            nneg_count += 1
    if neg_count > nneg_count:
        print('There are more negative returns than non-negative across 2018-2022 for TGT.\n')
    else:
        print('There are more non-negative returns than negative across 2018-2022 for TGT.\n')

# Year by year
def yearly_neg_or_nonneg(dataset):
    print("Comparing yearly returns across 2018-2022...")
    for year in years:
        year_neg_total = 0
        year_nneg_total = 0
        for weekday in weekdays: 
            neg_count = len(get_neg_returns(dataset, weekday, year))
            nneg_count = len(get_nonneg_returns(dataset, weekday, year))
            if neg_count > nneg_count:
                print('There are more negative returns for ' + year + ' ' + weekday + 's.')
            else:
                print('There are more non-negative returns for ' + year + ' ' + weekday + 's.')
            year_neg_total += neg_count
            year_nneg_total += nneg_count
        if year_neg_total > year_nneg_total:
            print('There are more negative returns for all of ' + year + '.\n')
        else:
            print('There are more non-negative returns for all of ' + year + '.\n')


neg_or_nonneg(tgt_file)
yearly_neg_or_nonneg(tgt_file)

# Question 1.4 - Calculate the impact of the negative and non-negative returns (which has greater movement)
print("Question 1.4 & 1.5")
def yearly_impact(dataset, year):
    # Calculate the mean of each weekday for the year 
    neg_year_means = [] # list of neg means for each day
    nneg_year_means = [] # list of nneg means for each day
    for weekday in weekdays:
        neg_mean = get_mean(get_neg_returns(dataset, weekday, year))
        nneg_mean = get_mean(get_nonneg_returns(dataset, weekday, year))
        if abs(neg_mean) > nneg_mean:
            print('TGT loses more on a "down" %s than gains on an "up" %s in %s' % (weekday, weekday, year))
        else:
            print('TGT gains more on an "up" %s than loses on a "down" %s in %s' % (weekday, weekday, year))
        neg_year_means.append(neg_mean)
        nneg_year_means.append(nneg_mean)
    #now calculate for the overall year
    neg_mean = get_mean(neg_year_means)
    nneg_mean = get_mean(nneg_year_means)
    if abs(neg_mean) > nneg_mean:
        print('TGT loses more on a "down" day than gains on an "up" day in %s\n' % (year))
    else:
        print('TGT gains more on an "up" day than loses on a "down" day in %s\n' % (year))


for year in years:
    yearly_impact(tgt_file,year)

# See attached Word Document for question 1.2 and additional supplementation.