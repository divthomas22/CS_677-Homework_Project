"""
Divya Thomas 
Class: CS 677
Date: 4/12/2023
Description of Problem (just a 1-2 line summary!): 
This file analyzes the impact of each of the features on the most accurate classifier (consolidated)
"""
import os
import initial_analysis as ia
import consolidated as c
import matplotlib.pyplot as plt

# function to remove a given feature from the dataset
def remove(train, test, f):
    xtrain = train[0].drop(f, axis=1)
    xtest = test[0].drop(f, axis=1)

    ftrain = (xtrain, train[1])
    ftest = (xtest, test[1])
    return(ftrain, ftest)


def plot_f_accuracy(train, test):
    features = ['audio_valence', 'acousticness', 'energy', 'instrumentalness', 'loudness']
    accuracies = []
    for feature in features:
        ftrain, ftest = remove(train, test, feature)
        pred = c.consolidated_classifier(ftrain, ftest)
        acc = ia.accuracy(pred)
        accuracies.append(acc)
    
    fig,ax = plt.subplots()
    ax.set_xlabel('Missing feature')
    ax.set_ylabel('Accuracy(%)')
    input_dir = os.path.abspath(__file__) + '\..\datasets'
    plot_file = os.path.join(input_dir, 'missing_feature.png')
    ax.scatter(features, accuracies)

    plt.savefig(plot_file)



df = ia.dataset()
split_data = ia.split(df)
plot_f_accuracy(split_data[0], split_data[1])






