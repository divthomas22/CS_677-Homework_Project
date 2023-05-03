"""
Divya Thomas 
Class: CS 677
Date: 4/24/2023
Homework Problem #2
Description of Problem (just a 1-2 line summary!):
Analyze the performance of the Naive Bayesian classifier on this same dataset to compare to SVM.
"""
import question1 as q1
from sklearn.naive_bayes import GaussianNB


df = q1.create_pandas_dataframe()
split_data = q1.split(df)

# Question 2.1 - Use your own classifier to predict the test data set 
print ("\n--Question 2.1--")

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

nb = predict_nb(split_data[0], split_data[1])
print("NB Classifier Results: \n", nb)
acc = q1.accuracy(nb)
cm = q1.compute_confusion_matrix(nb)
print("Accuracy: %.2f%%" % acc, "\nConfusion Matrix:\n", cm )

# Question 2.2 - Calculate the TP, FP, TN, FN, TPR, and TNR for each classifier
print ("\n--Question 2.2--")
# get TP, FP, TN, FN
def calc_label_accuracies(df):
    # get the list of true labels 
    true_list = df['Actual'].tolist()
    # get the list of the predicted labels
    pred_list = df["Prediction"].tolist()

    #set up counters for each scenario
    tp_count = 0
    fp_count = 0
    tn_count = 0
    fn_count = 0

    for i in range(len(pred_list)):
        if true_list[i] == '+':
            if pred_list[i] == '+': #true positive (true normal)
                tp_count += 1
            elif pred_list[i] == '-': #false negative
                fn_count += 1
        elif true_list[i]   == '-': 
            if pred_list[i] == '-': #true negative
                tn_count += 1
            elif pred_list[i] == '+': #false positive
                fp_count += 1
    count_list = [tp_count, fn_count, tn_count, fp_count]            
    return count_list

# function to calculate the True Positive Rate based off of the count list provided
def calc_tpr(count_list):
    # TPR = TP/(TP+FN)
    tp = count_list[0]
    fn = count_list[1]
    #based off of calc_label_accuracies function

    tpr = tp / (tp + fn)
    return tpr

# function to calculate the True Negative Rate based off of the count list provided
def calc_tnr(count_list):
    # TNR = TN/(TN+FP)
    tn = count_list[2]
    fp = count_list[3]
    #based off of calc_label_accuracies function

    tnr = tn / (tn + fp)
    return tnr

def print_counts(df):
    acc = q1.accuracy(df)
    counts = calc_label_accuracies(df)
    tpr = calc_tpr(counts)
    tnr = calc_tnr(counts)
    print("TP: %d \nFN: %d \nTN: %d \nFP: %d \nTPR: %.2f \nTNR: %.2f \nAccuracy: %.2f%%" % (counts[0], counts[1], counts[2], counts[3], tpr, tnr, acc))


lk = q1.linear_kernel(split_data[0], split_data[1])
gk = q1.gaussian_kernel(split_data[0], split_data[1])
pk = q1.poly_kernel(split_data[0], split_data[1])

print("\n--Linear Kernel--")
print_counts(lk)
print("\n--Gaussian Kernel--")
print_counts(gk)
print("\n--Polynomial Kernel--")
print_counts(pk)
print("\n--Naive Bayesian--")
print_counts(nb)
