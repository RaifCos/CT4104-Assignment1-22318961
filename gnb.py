# Python Script to run Gaussian Naïve Bayes Classification Model for the Wildfire Dataset.
# Name: Raif Costello
# Student ID: 22318961

# TODO: Hyperparameter processing for priors.

from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

def main(train, test, var_smoothing, priors):
    # Define Features and Model.
    features = ['temp', 'humidity', 'rainfall', 'drought_code', 'buildup_index', 'wind_speed']
    model = GaussianNB(var_smoothing=var_smoothing, priors=priors)

    # Set "fire" column to binary values.
    trainFire = (train['fire'] == 'yes').astype(int).values
    trainFeatures = train[features].values
    
    # Preprocessing with Scaler.
    scaler = StandardScaler()
    trainFeaturesScaled = scaler.fit_transform(trainFeatures)

    # Train Model.
    model.fit(trainFeaturesScaled, trainFire)

    # Prepare Testing Set like Training Set.
    testFire = (test['fire'] == 'yes').astype(int).values
    testFeatures = test[features].values
    # Preprocessing without Scaler.
    testFeaturesScaled = scaler.transform(testFeatures)

    # Calculate accuracy of training and testing sets.
    trainAccuracy = model.score(trainFeaturesScaled, trainFire)
    testAccuracy = model.score(testFeaturesScaled, testFire)

    # Print and return results.
    print(f"Training Accuracy: {trainAccuracy:.2f}")
    print(f"Testing Accuracy: {testAccuracy:.2f}")
    return testAccuracy