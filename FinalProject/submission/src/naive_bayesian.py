"""
Divya Thomas 
Class: CS 677
Date: 4/12/2023
Description of Problem (just a 1-2 line summary!): 
Use the Naive Bayesian classifier to predict the popularity of test data and colculate its accuracy. 
"""
import initial_analysis as ia
from sklearn.naive_bayes import GaussianNB

# Use Naive Bayesian NB classifier to predict class values for test data
def predict_nb(train, test):

    # Fit a Naive Bayesian classifier to the training data
    nb = GaussianNB()
    nb.fit(train[0], train[1])

    #predict test data classes with the model
    pred = nb.predict(test[0])

    #create a copy of the test data with the added predictions
    test_copy = test[0].copy()
    test_copy['Actual'] = test[1]
    test_copy['Prediction'] = pred

    return test_copy

def print_counts(prediction_data):
    # Calculate the TP, FP, TN, FN, TPR, and TNR 
    count_list = ia.calc_label_accuracies(prediction_data)
    acc = ia.accuracy(prediction_data)
    tpr = ia.calc_tpr(count_list)
    tnr = ia.calc_tnr(count_list)   
    print('-Naive Bayesian Counts-')
    print('TP: %d \nFP: %d \nTN: %d \nFN: %d \nAccuracy: %.2f \nTPR: %.2f \nTNR: %.2f' 
        % (count_list[0], count_list[3], count_list[2], count_list[1], acc, tpr, tnr))
    print("Confusion Matrix:\n", ia.compute_confusion_matrix(prediction_data))




