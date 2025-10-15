# Python Script to run either Classification Model for the Wildfire Dataset with user-defined Hyperparameters.
# Name: Raif Costello
# Student ID: 22318961

import pandas as pd

def main():
    waiting = True
    while waiting:
        model = input("Which model would you like to use, SVM or GNB? > ").lower()
        if model == "svm":
            callSVM()
            waiting = False
            break
        if model == "gnb":
            callGNB()
            waiting = False
            break
        else:
            print("Not a valid model!")

# Helper Function for users to define Hyperparameters in the correct range.
def defineHyperparameter(name, min, max):
    waiting = True
    while waiting:
        try:
            value = float(input(f"What value should the Hyperparameter {name} be? (between {min} and {max}) > "))
            if min <= value <= max:
                waiting = False
                return value
            else:
                print(f"Value must be between {min} and {max}.")
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

# Import Support Vector Machine Module and run.
def callSVM():
    # Hyperparemeters
    gamma = defineHyperparameter("gamma", 0, 1)
    C = defineHyperparameter("C", 0, 10)

    print(f"Running Support Vector Machine Classification Model...")
    import svm
    svm.main(trainingSet, testingSet, gamma, C)

# Import Gaussian Naïve Bayes and run.
def callGNB():
    # Hyperparemeters
    var_smoothing = defineHyperparameter("var_smoothing", 1e-10, 1e-5)
    fireProb = defineHyperparameter("Priors (\"fire\" - yes)", 0, 1)
    priors = [fireProb, 1 - fireProb]
    print(f"Running Gaussian Naïve Bayes Classification Model...")
    import gnb
    gnb.main(trainingSet, testingSet, var_smoothing, priors)

# Read data from CSV before starting main loop.
trainingSet = pd.read_csv('data/wildfires_training.csv')
testingSet = pd.read_csv('data/wildfires_test.csv')
main()