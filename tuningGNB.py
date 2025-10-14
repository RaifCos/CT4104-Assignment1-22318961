# Python Script to tune Hyperparameters for the GNB Classification Model for the Wildfire Dataset.
# Name: Raif Costello
# Student ID: 22318961

# TODO: Hyperparameter tuning for priors.

import gnb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

trainingSet = pd.read_csv('data/wildfires_training.csv')
testingSet = pd.read_csv('data/wildfires_test.csv')

# Choose steps to increment by.
stepVarSmoothing = 1e-10

# Iterate over Hyperparameter ranges to find the best combination.
varSmoothing = 1e-10

while varSmoothing < 1e-5:
    # Run GNB model with current Hyperparameters and retieve accuracy.
    acc = gnb.main(trainingSet, testingSet, varSmoothing, None)
    print(f"var_smoothing: {varSmoothing}, Accuracy: {acc:.2f}")
    varSmoothing += stepVarSmoothing