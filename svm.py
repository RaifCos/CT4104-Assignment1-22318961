# Python Script to run Support Vector Machine Classification Model for the Wildfire Dataset.
# Name: Raif Costello
# Student ID: 22318961

from sklearn import svm
from sklearn.preprocessing import StandardScaler

def main(train, test, gamma, C):
    # Define Features and Model.
    features = ['temp', 'humidity', 'rainfall', 'drought_code', 'buildup_index', 'wind_speed']
    model = svm.SVC(kernel="poly", gamma=gamma, C=C)

    # Set "fire" column to binary values.
    trainFire = (train['fire'] == 'yes').astype(int).values
    trainFeatures = train[features].values
    
    # Preprocessing and fitting data with Scaler.
    scaler = StandardScaler()
    trainFeaturesScaled = scaler.fit_transform(trainFeatures)

    # Train Model.
    model.fit(trainFeaturesScaled, trainFire)

    # Prepare Testing Set like Training Set.
    testFire = (test['fire'] == 'yes').astype(int).values
    testFeatures = test[features].values
    # Preprocessing again, now without fitting.
    testFeaturesScaled = scaler.transform(testFeatures)

    # Calculate accuracy of training and testing sets.
    trainAccuracy = model.score(trainFeaturesScaled, trainFire)
    testAccuracy = model.score(testFeaturesScaled, testFire)

    # Print and return results.
    return testAccuracy