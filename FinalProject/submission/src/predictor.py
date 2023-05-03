"""
Divya Thomas 
Class: CS 677
Date: 4/12/2023
Description of Problem (just a 1-2 line summary!): 
This is a file that contains a popularity predictor by taking in user input and using the most accurate
(consolidated) classifier to predict a songs popularity.
"""
import pandas as pd
import numpy as np
import initial_analysis as ia
import consolidated as c

def get_prediction(values_dict):
    # user input is test data
    xtest = pd.DataFrame(values_dict, index=[0])
    ytest = [np.nan] # NAN value since there was no actual
    test = (xtest, ytest)

    # use the full dataset as training data now since
    df = ia.dataset()
    xtrain = df[['audio_valence', 'acousticness', 'energy', 'instrumentalness', 'loudness']]
    ytrain = df['rating']

    train = (xtrain, ytrain)

    pred_df = c.consolidated_classifier(train, test)

    # get the predicted value
    prediction = pred_df['Prediction'].iloc[0]

    return prediction


# request user input to predict a song's popularity
def ask_predictor():

    print('\nStarting up the song popularity predictor...')
    # a dictionary containing all tested feature values
    responses = {}
    features = ['audio_valence', 'acousticness', 'energy', 'instrumentalness', 'loudness'] #list of features

    for feature in features:
        response = input("Please input the song's %s score: " % feature)
        # add the response to the dictionary
        responses[feature] = float(response)
    
    print("Predicting...")

    prediction = get_prediction(responses)

    if (prediction == '+'):
        print("Congrats! Your song is predicted to be popular!")
    elif (prediction == '-'):
        print("Sorry, your song is predicted to be unsuccesful.")
    else: 
        print("Error: Unknown prediction")

ask_predictor()
