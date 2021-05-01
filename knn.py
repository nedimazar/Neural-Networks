from readARFF import getARFFData
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
np.random.seed(42)

# The Euclidean distance for two vectors
def dist(x, y):
    return (np.sqrt(abs(sum([(float(a) - float(b)) ** 2 for a, b in zip(x, y)]))))

def getScore(df, instance):
    distanceSum = sum([dist(row, instance) for index, row in df.iterrows()])

    # A smaller total distance should get a higher score
    return 1 / distanceSum

def majorityClassify(df, instance):
    scores = []
    for classification in df['class'].unique():
        if classification:
            subset = df[df['class'] == classification].drop(['class'], axis=1)
            scores.append((getScore(subset, instance), classification))
    return sorted(scores)[-1][1]

def classify(x, y):
    x.dropna(inplace=True)
    y.dropna(inplace=True)
    classes = []
    for row, instance in y.iterrows():
        classes.append(majorityClassify(x, instance))
    return classes

def getSplits(data, n = 5):
    splits = []
    chunks = np.array_split(data, n)

    for x in chunks:
        splits.append((pd.concat([y for y in chunks if not x is y]), x))
    
    return splits

def trainAndValidate(splits):
    overallAcc = 0.0

    for train, test in splits:
        predictions = classify(train, test)

        accuracy = accuracy_score(test['class'], predictions)

        print(f"Accuracy for current split: {accuracy}")

        overallAcc = overallAcc + (accuracy * 1/len(splits))
    print(f"Overall accuracy fo all splits: {overallAcc}")

if __name__ == '__main__':
    wine, wineAttr = getARFFData('wine.arff')
    iris, irisAttr = getARFFData('iris.arff')

    print("Doing 5 fold cross validation with the IRIS data: ")
    irisSplits = getSplits(iris)
    trainAndValidate(irisSplits)
    print()

    print("Doing 5 fold cross validation with the WINE data: ")
    wineSplits = getSplits(wine)
    trainAndValidate(wineSplits)
    