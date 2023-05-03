"""
Divya Thomas 
Class: CS 677
Date: 4/12/2023
Description of Problem (just a 1-2 line summary!): 
Use the k-NN classifier to predict the popularity of test data and colculate its accuracy. 
"""
import os 
import matplotlib.pyplot as plt
import initial_analysis as ia
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

def knn_classifier(train, test, k):
    # hold all accuracy values for each k
    accuracy = []
    df_list = []
    df = test[0].copy()
    df['Actual'] = test[1].copy()

    # scale the training and test data
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train[0])
    test_scaled = scaler.transform(test[0])

    for i in k:
        #classifier setup with k 
        classifier = KNeighborsClassifier(n_neighbors=i) 

        #train the classifier with the training set
        classifier.fit(train_scaled, train[1])


        #predict data using the classifier
        pred = classifier.predict(test_scaled)

        df_pred = df.copy()
        df_pred['Prediction'] = pred

        accuracy.append(ia.accuracy(df_pred))
        df_list.append(df_pred)

    return (accuracy, df_list)

def plot_accuracy(k, accuracy):
    fig,ax = plt.subplots()
    ax.set_xlabel('k')
    ax.set_ylabel('Accuracy(%)')
    input_dir = os.path.abspath(__file__) + '\..\datasets'
    plot_file = os.path.join(input_dir, 'knn_plot.png')
    ax.plot(k, accuracy)

    plt.savefig(plot_file)

def print_counts(prediction_data):
    # Calculate the TP, FP, TN, FN, TPR, and TNR of most accurate k (11)
    count_list = ia.calc_label_accuracies(prediction_data)
    acc = ia.accuracy(prediction_data)
    tpr = ia.calc_tpr(count_list)
    tnr = ia.calc_tnr(count_list)   
    print('-k-NN (k=11) Counts-')
    print('TP: %d \nFP: %d \nTN: %d \nFN: %d \nAccuracy: %.2f \nTPR: %.2f \nTNR: %.2f' 
        % (count_list[0], count_list[3], count_list[2], count_list[1], acc, tpr, tnr))
    print("Confusion Matrix:\n", ia.compute_confusion_matrix(prediction_data))

df= ia.dataset()
split_data = ia.split(df)
k_list = [3, 5, 7, 9, 11]
prediction_data = knn_classifier(split_data[0], split_data[1], k_list)
plot_accuracy(k_list, prediction_data[0])

