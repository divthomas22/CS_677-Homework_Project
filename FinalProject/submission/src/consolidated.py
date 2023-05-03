"""
Divya Thomas 
Class: CS 677
Date: 4/12/2023
Description of Problem (just a 1-2 line summary!): 
This is a file that contains a classifier that takes predictions of the four classifiers tested 
and makes a prediction based off of the most common prediction. 
"""
import initial_analysis as ia
import knn
import l_kernel_svm as lk
import logistic_regression as lr
import naive_bayesian as nb
import statistics as s

# Use consolidated classifier to calculate predictions from each classifier under the same split
def consolidated_classifier(train, test):
    # set of a dataframe to hold all test data predictions
    c_df = test[0].copy()
    c_df['Actual'] = test[1]

    # knn where k = 11 ( best k ) 
    knn_pred = knn.knn_classifier(train, test, [11]) 
    c_df['KNN'] = knn_pred[1][0]['Prediction']

    lk_pred = lk.linear_kernel(train, test)
    c_df['LK'] = lk_pred['Prediction']

    lr_pred = lr.lr_classifier(train, test)
    c_df['LR'] = lr_pred['Prediction']

    nb_pred = nb.predict_nb(train, test)
    c_df['NB'] = nb_pred['Prediction']

    # set up a list to store ensemble Prediction labels to
    e = []
    # iterate through the dataframe
    for index,row in c_df.iterrows():
        label_list = [] # create a list of all the labels
        label_list.append(row["KNN"])
        label_list.append(row["LK"])
        label_list.append(row["LR"])
        label_list.append(row["NB"])
        try:
            mode = s.mode(label_list)
        except s.StatisticsError:
            # if there are equal counts of each use default probability for training set
            mode = ia.get_default_probability(train[1])
        e.append(mode) # add the mode to the we list
    #now add the consolidated prediction list to the dataframe
    c_df["Prediction"] = e

    return c_df



def print_counts(prediction_data):
    # Calculate the TP, FP, TN, FN, TPR, and TNR 
    count_list = ia.calc_label_accuracies(prediction_data)
    acc = ia.accuracy(prediction_data)
    tpr = ia.calc_tpr(count_list)
    tnr = ia.calc_tnr(count_list)   
    print('-Consolidated Counts-')
    print('TP: %d \nFP: %d \nTN: %d \nFN: %d \nAccuracy: %.2f \nTPR: %.2f \nTNR: %.2f' 
        % (count_list[0], count_list[3], count_list[2], count_list[1], acc, tpr, tnr))
    print("Confusion Matrix:\n", ia.compute_confusion_matrix(prediction_data))


