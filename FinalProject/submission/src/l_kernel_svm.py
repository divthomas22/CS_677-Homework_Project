"""
Divya Thomas 
Class: CS 677
Date: 4/12/2023
Description of Problem (just a 1-2 line summary!): 
Use the linear kernel SVM to predict the popularity of test data and colculate its accuracy. 
"""
import initial_analysis as ia
from sklearn.svm import SVC

# Use linear kernel SVM classifier to predict class values for test data
def linear_kernel(train, test):

    # Fit a linear kernel SVM classifier to the training data
    lk_svm = SVC(kernel='linear')
    lk_svm.fit(train[0], train[1])

    #predict test data classes with the model
    pred = lk_svm.predict(test[0])

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
    print('-Linear-Kernel SVM Counts-')
    print('TP: %d \nFP: %d \nTN: %d \nFN: %d \nAccuracy: %.2f \nTPR: %.2f \nTNR: %.2f' 
        % (count_list[0], count_list[3], count_list[2], count_list[1], acc, tpr, tnr))
    print("Confusion Matrix:\n", ia.compute_confusion_matrix(prediction_data))



