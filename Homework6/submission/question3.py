"""
Divya Thomas 
Class: CS 677
Date: 4/24/2023
Homework Problem #3
Description of Problem (just a 1-2 line summary!):
Use k-means clustering to calculate distortions, plot clusters and create and analyze the performance
or a new classifier that predicts the class labels using euclidean distances to the centroids of the clusters.
"""
import os
import random
import math
import pandas as pd
import numpy as np
import question1 as q1 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# Question 3.1 - Use k-means clustering to calculate and plot distortions and find the best 'k'
print ("\n--Question 3.1--")
#Read the data into the dataframe taking in all the L values
#function to convert the csv data into a pandas dataframe
def create_pandas_dataframe():
    #get file path
    input_dir = os.path.abspath(__file__) + '\..\datasets'
    file = os.path.join(input_dir, 'seeds_dataset.csv')

    columns = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'L']

    #convert to df
    full_df = pd.read_csv(file, names=columns, delimiter='\t') 

    return full_df

df = create_pandas_dataframe()
print(df)

# function to calculate and plot distortions for k in range 1-8
def plot_kmeans(df, name):
    # get file location 
    input_dir = os.path.abspath(__file__) + '\..\datasets'
    file = os.path.join(input_dir, name)

    # Initialize list to store distortions for each value of k
    distortions = []

    # Compute distortion for each value of k
    for k in range(1, 9):
        kmeans = KMeans(n_clusters=k, init='random', n_init=10)
        kmeans.fit(df)
        distortions.append(kmeans.inertia_)

    plt.plot(range(1, 9), distortions)
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.savefig(file)
    print("Plot saved to :" + file)

plot_kmeans(df, 'distortions.png')




# Question 3.2 - Re-run the clustering on the best k (3) and
# plot the clusters and centroids for two random features
print ("\n--Question 3.2--")

# function to calculate distortions for a given k
def kmeans_clusters(df, k, fi, fj, name):
    # get file location 
    input_dir = os.path.abspath(__file__) + '\..\datasets'
    file = os.path.join(input_dir, name)

    # if we are choosing the features randomly, these params will be 0, awaiting to be assigned 
    # otherwise we will move forward with whatever is passed 
    if (fi == 0 & fj == 0):
        features = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7']
        # Pick two random features from the list
        fi, fj = random.sample(features, k=2)
    # otherwise use the values provided
    else: 
        fi = 'f'+ str(fi)
        fj = 'f'+ str(fj)

    # Compute distortion for each value of k
    kmeans = KMeans(n_clusters=k, init='random', n_init=10)
    kmeans.fit(df[[fi, fj]])
    fig, ax = plt.subplots()

    # Plot the data points with different colors for each cluster
    for label in np.unique(kmeans.labels_):
        ix = np.where(kmeans.labels_ == label)
        ax.scatter(df.loc[ix, fi], df.loc[ix, fj], label="Cluster " + str(label), alpha=0.7)

    # Plot the centroids with black markers
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='black', label='Centroids')
     #display legend and labels
    ax.legend()
    ax.set_xlabel(fi)
    ax.set_ylabel(fj)
    plt.savefig(file) # save the figure
    print ("Plot for %s and %s saved to %s" % (fi, fj, file))

    # create a copy of the original dataframe with the cluster labels added to it 
    cluster_df = df.copy()
    cluster_df['Cluster'] = kmeans.labels_

    # return both the cluster_df and the centroids 
    return (cluster_df, kmeans.cluster_centers_)


kmeans_clusters(df, 3, 0, 0, 'clusters_3_2.png')

# Question 3.3 - Re-run the clustering on the best k (3) and the features from before (f5 and f4)
# list the centroids and the assigned class to each cluster based off of the majority 
print ("\n--Question 3.3--")

# To get majority class for each cluster
def get_cluster_class(df):

    # get unique cluster values from the dataframe
    clusters = df['Cluster'].unique()
    #dict to hold values as cluster no as the key and class as the value
    classes = {}
    for cluster in clusters:
        sub_df = df[df["Cluster"] == cluster]
        counts = []
        for i in range(1,4):
            count = len(sub_df[sub_df["L"] == i])
            counts.append(count)
        max_val = max(counts)
        max_class = counts.index(max_val) + 1
        classes['c'+ str(cluster)] = max_class

    return classes

# this time retrieve return data 
cluster_df , centroids = kmeans_clusters(df, 3, 5, 4, 'clusters_3_3.png')
print("\nCluster Dataframe for k=3:\n", cluster_df)

classes = get_cluster_class(cluster_df)
for cluster in range(0,3):
    print("Cluster: %d - Centroid: (%.2f, %.2f) - Class: %d" 
          % (cluster, centroids[cluster][0], centroids[cluster][1], classes.get('c' + str(cluster))))


# Question 3.4 - From the clustering on the best k (3) and the features from before (f5 and f4), 
# create a multi-label classifier that calculates the euclidean distance between each centroid and 
# predicts the class based off of the closest.
print ("\n--Question 3.4--")

# to predict the class based off of euclidean distances
def predict_ed(df, class_dict, centroids, fi, fj):

    #list of predictions 
    pred = []

    fi = 'f' + str(fi)
    fj = 'f' + str(fj)

    x_list = df[fi].tolist()
    y_list = df[fj].tolist()

    #iterate through
    for i in range(0, len(x_list)):
        distances = []
        for c in range(0, 3): # each cluster (0, 1, 2) 
            #calculate the distance from the point to each centroid 
            ed = calc_ed((x_list[i], y_list[i]), centroids[c])
            distances.append(ed)
        # get the index of the min distance to get which cluster's centroid is closest
        cluster = distances.index(min(distances))

        # get the class value associated with the cluster
        label = class_dict['c'+ str(cluster)]

        # add the label to the list of predictions
        pred.append(label)

    # add the prediction column to the dataframe and rename 'L' to 'Actual' column
    pred_df = df.copy()
    pred_df = pred_df.rename(columns={'L': 'Actual'})
    pred_df['Prediction'] = pred 

    return pred_df
        

def calc_ed(point1, point2):
    # calculate the euclidean distance between point1 and point2
    distance = math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    return distance

pred_df = predict_ed(df, classes, centroids, 5, 4)
print(pred_df)

acc = q1.accuracy(pred_df)
print("The accuracy of this classifier on this entire dataset is %.2f%%" % acc)



# Question 3.5 - From the prediction dataframe, remove any rows containing 2 as the Actual or Prediction label
# (since in questions 1 and 2, I only used labels 1 and 3)
# and calculate the accuracy of these rows and confusion matrix

print ("\n--Question 3.5--")

# to remove label=2 rows 
def remove_label(df):
    df = df.loc[(df['Actual'] != 2) & (df['Prediction'] != 2)]
    return df 

removed_df = remove_label(pred_df)
print("Prediction Dataframe after removing label 2 rows: \n", removed_df)

acc = q1.accuracy(removed_df)
cm = q1.compute_confusion_matrix(removed_df)

print("Accuracy: %.2f%%" % acc)
print("Confusion Matrix:\n", cm)