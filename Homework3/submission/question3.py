"""
Divya Thomas 
Class: CS 677
Date: 4/1/2023
Homework Problem #3
Description of Problem (just a 1-2 line summary!):
Use the k-NN classifier to predict test data classes and evaluate its efficiency.
"""
import os
import question1 as q1
import question2 as q2
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

bn_df = q1.create_pandas_dataframe()

print ("\n--Question 3.1--")
# Quesiton 3.1 - Use a k-NN classifier to train and predict test data
#function to create the test and training data with a dataframe
def create_test_train(df):
    x = df[['f1', 'f2', 'f3', 'f4']]
    y = df['color']
    #split dataset 50/50 into a training and testing set 
    x_train,x_test,y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)

    train = (x_train, y_train)
    test = (x_test, y_test)
    return (train, test)

def knn_classifier(train, test, k):
    # hold all accuracy values for each k
    accuracy = []
    df_list = []
    df = test[0].copy()
    df['color'] = test[1].copy()

    for i in k:
        #classifier setup with k 
        classifier = KNeighborsClassifier(n_neighbors=i) 

        #train the classifier with the training set
        classifier.fit(train[0], train[1])


        #predict data using the classifier
        pred = classifier.predict(test[0])

        df_pred = df.copy()
        df_pred['prediction'] = pred

        accuracy.append(q2.accuracy(df_pred))
        df_list.append(df_pred)

    return (accuracy, df_list)

train, test = create_test_train(bn_df)
k_list = [3, 5, 7, 9, 11]
prediction_data = knn_classifier(train, test, k_list)
print("k", "accuracy")
for i in range(0, len(k_list)):
    print(k_list[i], prediction_data[0][i])


# Question 3.2 - Plot the accuracies for each  k
print ("\n--Question 3.2--")
def plot_accuracy(k, accuracy):
    fig,ax = plt.subplots()
    ax.set_xlabel('k')
    ax.set_ylabel('Accuracy(%)')
    input_dir = os.path.abspath(__file__) + '\..\datasets'
    plot_file = os.path.join(input_dir, 'accuracy_plot.pdf')
    ax.plot(k, accuracy)

    plt.savefig(plot_file)

plot_accuracy(k_list, prediction_data[0])
print("See plot in datasets/accuracy_plot.pdf")

# Question 3.3 - Calculate the TP, FP, TN, FN, TPR, and TNR of most accurate k (7)
print ("\n--Question 3.3--")
count_list = q2.calc_label_accuracies(prediction_data[1][2])
tpr = q2.calc_tpr(count_list)
tnr = q2.calc_tnr(count_list)

print('-Counts-')
print('TP: %d \nFP: %d \nTN: %d \nFN: %d \nAccuracy: %.2f \nTPR: %.2f \nTNR: %.2f' 
      % (count_list[0], count_list[3], count_list[2], count_list[1], prediction_data[0][2], tpr, tnr))

# Question 3.4 - See supplemental documentation

# Question 3.5 - Predict the class label for a bill with the last four digits of my BUID (5732)
# as feature values using both the simple and optimal k-nn classifiers
print ("\n--Question 3.5--")

def predict_bill(f1, f2, f3, f4, train_data):
    # create a dictionary containing the bill to be tested 
    bill_data = {'f1': [f1],'f2': [f2],'f3': [f3],'f4': [f4]}
    #create dataframe from this data 
    df = pd.DataFrame(bill_data)

    #use the simple classifier
    df_simple = q2.predict_validity(df.copy())
    print("Simple Classifier: \n", df_simple)

    #use the k-nn classifier for k = 7
    classifier = KNeighborsClassifier(n_neighbors=7) 

    #train the classifier with the training set
    classifier.fit(train_data[0], train_data[1])

    #predict data using the classifier
    pred = classifier.predict(df)

    df_knn = df.copy()
    df_knn['prediction'] = pred
    print("7-NN Classifier: \n", df_knn)

    return (df_simple, df_knn)



predict_bill(5,7,3,2,train)