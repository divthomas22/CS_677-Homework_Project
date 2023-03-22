"""
Divya Thomas 
Class: CS 677
Date: 3/18/2023
Homework Problem #2
Description of Problem (just a 1-2 line summary!): 
Examination of the tables created in problem 1 and further analyses.
"""
import question1 as q1
import read_stock_data_from_file as rdf

# Get the dataset for 'TGT'
tgt_file =rdf.create_dataset('TGT')

# List of weekdays and years to get return results for
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
years = ['2018', '2019', '2020', '2021', '2022']

# Question 2.1-2,4 : Answered on supplementary word document

# Question 2.3 - Use the previous question's functions to determine the best 
# and worst weekday returns of the year.
print("Question 2.3")
def get_best_weekday(dataset, year):
    best_mean = -1
    best_weekday = ''
    for weekday in weekdays:
        mean = q1.get_mean(q1.get_all_returns(dataset, weekday, year))
        if mean > best_mean:
            best_mean = mean
            best_weekday = weekday
    print(year + " Best Weekday: " + best_weekday )

def get_worst_weekday(dataset, year):
    worst_mean = 1
    worst_weekday = ''
    for weekday in weekdays:
        mean = q1.get_mean(q1.get_all_returns(dataset, weekday, year))
        if mean < worst_mean:
            worst_mean = mean
            worst_weekday = weekday
    print(year + " Worst Weekday: " + worst_weekday )


for year in years:
    get_best_weekday(tgt_file, year)
    get_worst_weekday(tgt_file, year)
    print('\n')

