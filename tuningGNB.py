# Python Script to tune Hyperparameters for the GNB Classification Model for the Wildfire Dataset.
# Name: Raif Costello
# Student ID: 22318961

import gnb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

trainingSet = pd.read_csv('data/wildfires_training.csv')
testingSet = pd.read_csv('data/wildfires_test.csv')

# Choose steps to increment by.
stepVarSmoothing = 1e-5/100
stepPriors = 0.001

# Lists to store results.
valuesVarSmoothing = []
valuesFirePrior = []
valuesAccuracy = []

# Iterate over Hyperparameter ranges to find the best combination.
varSmoothing = 1e-10
while varSmoothing < 1e-5:
    firePrior = 0.001
    while firePrior <= 0.9:
        # Round to avoid floating point errors.
        roundedFirePrior = round(firePrior, 3)
        roundedNoFirePrior = round(1 - firePrior, 3)
        priors = [roundedFirePrior, roundedNoFirePrior]

        # Run GNB model with current Hyperparameters and retieve accuracy.
        acc = gnb.main(trainingSet, testingSet, varSmoothing, priors)
        print(f"var_smoothing: {varSmoothing:.10f}, \"fire\" Prior: [{roundedFirePrior}, {roundedNoFirePrior}], Accuracy: {acc:.2f}")

        # Store results
        valuesFirePrior.append(firePrior)
        valuesVarSmoothing.append(varSmoothing)
        valuesAccuracy.append(acc)

        firePrior +=stepPriors
    varSmoothing += stepVarSmoothing

# Create a Pivot Table to create Heatmap.
countVarSmoothing = len(set(valuesVarSmoothing))
countFirePrior = len(set(valuesFirePrior))
pivotTable = np.array(valuesAccuracy).reshape(countVarSmoothing, countFirePrior)

# Create a Heatmap to display results.
plt.figure(figsize=(12, 8))
sns.heatmap(pivotTable, annot=False, cmap='viridis', cbar_kws={'label': 'Accuracy'})
plt.title('GNB Hyperparameter Tuning Results', fontsize=16, pad=20)
plt.xlabel('Priors [X, 1-X]', fontsize=12)
plt.ylabel('var_smoothing', fontsize=12)
plt.tight_layout()
plt.show()

# Find best Hyperparameters.
bestResult = valuesAccuracy.index(max(valuesAccuracy))
bestFirePrior = valuesFirePrior[bestResult]
bestVarSmoothing = valuesVarSmoothing[bestResult]
bestAccuracy = valuesAccuracy[bestResult]

# Print Results.
print(f"Best Parameters:")
print(f"var_smoothing: {bestVarSmoothing:.10f}")
print(f"Priors: [{bestFirePrior:.3f}, {1 - bestFirePrior:.3f}]")
print(f"Accuracy: {bestAccuracy:.2f}")