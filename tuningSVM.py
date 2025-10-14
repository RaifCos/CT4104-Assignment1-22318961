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
        # Round to avoid floating point errors.
        roundedGamma = round(gamma, 1)
        roundedC = round(c, 2)
        
        # Run SVM model with current Hyperparameters and retieve accuracy.
        acc = svm.main(trainingSet, testingSet, roundedGamma, roundedC)
        print(f"Gamma: {roundedGamma:.1f}, C: {roundedC:.2f}, Accuracy: {acc:.2f}")
        
        # Store results
        valuesGamma.append(roundedGamma)
        valuesC.append(roundedC)
        valuesAccuracy.append(acc)
        
        c += stepC
    gamma += stepGamma

# Create a Pivot Table to create Heatmap.
countGamma = len(set(valuesGamma))
countC = len(set(valuesC))
pivotTable = np.array(valuesAccuracy).reshape(countGamma, countC)

# Create a Heatmap to display results.
plt.figure(figsize=(12, 8))
sns.heatmap(pivotTable, annot=False, cmap='viridis', cbar_kws={'label': 'Accuracy'})
plt.title('SVM Hyperparameter Tuning Results', fontsize=16, pad=20)
plt.xlabel('C Parameter', fontsize=12)
plt.ylabel('Gamma Parameter', fontsize=12)
plt.tight_layout()
plt.show()

# Find best Hyperparameters.
bestResult = valuesAccuracy.index(max(valuesAccuracy))
bestGamma = valuesGamma[bestResult]
bestC = valuesC[bestResult]
bestAccuracy = valuesAccuracy[bestResult]

# Print Results.
print(f"Best Parameters:")
print(f"Gamma: {bestGamma:.1f}")
print(f"C: {bestC:.2f}")
print(f"Accuracy: {bestAccuracy:.2f}")