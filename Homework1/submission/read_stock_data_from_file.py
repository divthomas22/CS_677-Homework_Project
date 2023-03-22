"""
Divya Thomas 
Class: CS 677
Date: 3/18/2023
Homework Problem # Preliminary
Description of Problem (just a 1-2 line summary!):
This is file that will be referenced in other questions to call the create_dataset 
function. This function reads the lines from the ticker file argument and creates 
a list containing a dict of each stock record.
"""
import os

def create_dataset(ticker):
    input_dir = os.path.abspath(__file__) + '\..\datasets'
    ticker_file = os.path.join(input_dir, ticker + '.csv')

    headers = []
    data = []

    try:   
        with open(ticker_file) as f:
            lines = f.read().splitlines()
        print('opened file for ticker: ', ticker)
        #counting the lines to be read
        count = 1
    
    except Exception as e:
        print(e)
        print('failed to read stock data for ticker: ', ticker)

    #iterate over each line
    for line in lines:
        #split the line on a comma delimiter into an array of each value
        line = line.split(',')
        if count == 1: #header row
            headers = line
        else:
            #create a dict with headers as the key and values in this line
            record_dict = {}
            for i in range(len(line)):
                record_dict[headers[i]] = line[i]

            # add this line's dict to the dataset
            data.append(record_dict)
        count += 1
    
    return data

# test data pull for each ticker's data
tgt_dataset = create_dataset('TGT')
spy_dataset = create_dataset('SPY')

#output record count for testing
print('TGT record count: ',len(tgt_dataset))
print('SPY record count: ',len(spy_dataset))












