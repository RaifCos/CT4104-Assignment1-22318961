# Python Script to tune Hyperparameters for the SVM Classification Model for the Wildfire Dataset.
# Name: Raif Costello
# Student ID: 22318961

import svm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

trainingSet = pd.read_csv('data/wildfires_training.csv')
testingSet = pd.read_csv('data/wildfires_test.csv')

# Choose steps to increment by.
stepC = 0.01
stepGamma = 0.1

# Lists to store results.
valuesGamma = []
valuesC = []
valuesAccuracy = []

# Iterate over Hyperparameter ranges to find the best combination.
gamma = 0
while gamma < 10:
    c = 0.1
    while c < 1:
        # Eliminate Rounding Errors
        gammaRounded = round(gamma, 2)
        cRounded = round(c, 2)
        
        acc = svm.main(trainingSet, testingSet, gammaRounded, cRounded)
        print(f"Gamma: {gammaRounded}, C: {cRounded}, Accuracy: {acc:.4f}")
        
        # Store results
        valuesGamma.append(gammaRounded)
        valuesC.append(cRounded)
        valuesAccuracy.append(acc)
        
        c += stepC
    gamma+= stepGamma

# Create a Pivot Table to Graph results.
countC = len(set(valuesGamma))
countGamma = len(set(valuesC))
pivotTable = np.array(valuesAccuracy).reshape(countGamma, countC)

# Create a heatmap to represent results.
plt.figure(figsize=(12, 8))
sns.heatmap(pivotTable, annot=False, cmap='viridis', cbar_kws={'label': 'Accuracy'})
plt.title('SVM Hyperparameter Tuning Results', fontsize=16, pad=20)
plt.xlabel('C', fontsize=12)
plt.ylabel('Gamma', fontsize=12)
plt.tight_layout()
plt.show()

# Find and print best parameters.
bestResult = results_df['accuracy'].idxmax()
bestGamma = results_df.loc[bestResult, 'gamma']
bestC = results_df.loc[bestResult, 'C']
bestAccuracy = results_df.loc[bestResult, 'accuracy']

print(f"\nBest Parameters:")
print(f"Gamma: {bestGamma}")
print(f"C: {bestC}")
print(f"Accuracy: {bestAccuracy}%")