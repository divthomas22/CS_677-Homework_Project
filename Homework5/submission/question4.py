"""
Divya Thomas 
Class: CS 677
Date: 4/15/2023
Homework Problem #4
Description of Problem (just a 1-2 line summary!):
Perform predictions of the NSP_CLASS using the Random Forest classifier
and analyze its performance.
"""
import os
import question2 as q2
import question3 as q3
import question5 as q5
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# Question 4.1 - Split df up into train and test data (X:['MSTV', 'Width', 'Mode', 'Variance', 'NSP'], Y : NSP_CLASS)
# then use Random Forest classifier to predict class values for test data 
print ("\n--Question 4.1--")

# Random Forest classifier to predict class values for test (n - number of subtrees, d - max depth)
def predict_rf(train, test, n, d):

    # Fit a Decision Tree classifier to the training data
    rf = RandomForestClassifier(n_estimators=n, max_depth=d)
    rf.fit(train[0], train[1])

    #predict test data classes with the model
    pred = rf.predict(test[0])

    #create a copy of the test data with the added predictions
    test_copy = test[0].copy()
    test_copy['Actual'] = test[1]
    test_copy['Prediction'] = pred

    return test_copy

# calulate the error rate given a prediction df 
def error_rate(df):
    #calculate the accuracy of the predictions 
    acc = q2.accuracy(df)
    #get the error rate by (1 - (acc/100))
    e = 1 - (acc/100)
    return e

# calculate all prediction combination error rates are store them in a df 
def compute_error_df(train, test):
    # create an empty dataframe to hold error rates 
    df = pd.DataFrame(columns=['N', 'd', 'Error'])
    rows = []

    for n in range(1, 11):
        for d in range(1,6):
            #get the df that holds the predictions with these n and d values
            pred_df = predict_rf(train, test, n, d)
            #calculate the error rate for the prediction
            e = error_rate(pred_df)
            #append add the row to the list of rows 
            new_row = {'N': n, 'd': d, 'Error': e}
            rows.append(new_row)

    # add all new rows into the dataframe 
    df = pd.concat([df, pd.DataFrame(rows)])
    return df 


#get train and test data from q2
split_data = q2.get_split()
#dataframe with all error rates 
error_df = compute_error_df(split_data[0], split_data[1])
print("Error Rates: \n", error_df)




# # Question 4.2 - Plot the error rates for each combination
print ("\n--Question 4.2--")

def plot_err_rates(df, filename):
    #get file path to save plot to
    input_dir = os.path.abspath(__file__) + '\..\datasets'
    file = os.path.join(input_dir, filename)

    for d in range(1, 6):
        # get all rows with d column = d
        d_df = df[df['d'] == d]

        n = d_df['N'].tolist()
        rates = d_df['Error'].tolist()

        label = 'd='+str(d)
        plt.plot(n, rates, label=label) #plot the rates for the d value

    plt.xlabel('N')
    plt.ylabel('Error Rate')
    plt.legend() #add plot legend

    plt.savefig(file) #save the plot

    print("Error Rates plot saved to " + file)

plot_err_rates(error_df, "error_rates.png")

# Question 4.3 - Compute the accuracy for the best combination of N and d
print ("\n--Question 4.3--")

def calc_best_accuracy(df):
    # Find the row with the lowest value in the 'Error' column
    min_index = df['Error'].idxmin()
    row = df.loc[min_index]

    n = row['N']
    d = row['d']
    e = row['Error']

    acc = (1 - e)*100
    return [n, d, acc]

best = calc_best_accuracy(error_df)
print("Accuracy of Random Forest (N=%d, d=%d) is %.2f%%" % (best[0], best[1], best[2]))

# Question 4.4 - Compute the confusion matrix for the most accurate combination
print ("\n--Question 4.4--")

def compute_cm(train, test, n, d):
    #get the df that holds the predictions with these n and d values
    pred_df = predict_rf(train, test, n, d)
    cm = q2.compute_confusion_matrix(pred_df)
    return (pred_df, cm)

results, cm = compute_cm(split_data[0], split_data[1], best[0], best[1])
print("Confusion Matrix for Random Forest (N=%d, d=%d): \n" % (best[0], best[1]), cm)

# For Question 5
counts = q5.calc_label_accuracies(results)
tpr = q5.calc_tpr(counts)
tnr = q5.calc_tnr(counts)
def print_counts():
    print("\n--RF Counts (Q5)--")
    print("TP: %d \nFN: %d \nTN: %d \nFP: %d \nTPR: %.2f \nTNR: %.2f \nAccuracy: %.2f%%" % (counts[0], counts[1], counts[2], counts[3], tpr, tnr, best[2]))

print_counts()