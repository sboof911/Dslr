import numpy as np
import pandas as pd
import json, csv

def SigmoidFunction(y):
    return 1 / (1 + np.exp(-y))

def forward_propagation(features : np.ndarray, Weight : np.ndarray, Bias : np.ndarray):
    z = np.dot(features, Weight) + Bias
    y_pred = SigmoidFunction(z)
    return y_pred

def predict(features_values, Weights, Bias):
    y_pred = forward_propagation(features_values, Weights, Bias)
    return np.argmax(y_pred, axis=1)

def normilizedata(df : pd.DataFrame, means, stds):
    df = df.select_dtypes(include='number')
    df.drop('Index', axis=1, inplace=True)
    df.dropna(axis=1, inplace=True, how='all')
    NormalizedDF = df.copy()
    for col in df.columns:
        if col in means:
            NormalizedDF[col] = NormalizedDF[col].fillna(means[col])
            NormalizedDF[col] = (df[col] - means[col]) / stds[col]
    return NormalizedDF

def get_predictions(df : pd.DataFrame, predictData : dict):
    df = normilizedata(df, predictData["means"], predictData["stds"])
    y_pred = predict(df.values, predictData["wights"], predictData["bias"])
    HouseToInt = {'Ravenclaw': 0, 'Slytherin': 1, 'Gryffindor': 2, 'Hufflepuff': 3}
    IntToHouse = {v: k for k, v in HouseToInt.items()}
    y_pred_names = np.vectorize(IntToHouse.get)(y_pred)
    with open("houses.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Index', 'Hogwarts House'])
        for idx, house in enumerate(y_pred_names):
            writer.writerow([idx, house])

if __name__ == "__main__":
    import sys, os

    try:
        if len(sys.argv) != 2:
            raise Exception("Enter just the filepath as an arguments!")
        if not sys.argv[1].endswith(".csv"):
            raise Exception(f"The file '{sys.argv[1]}' must be a csv file.")
        if not os.path.exists(sys.argv[1]):
            raise Exception(f"The file '{sys.argv[1]}' does not exist.")
        if not os.path.exists("predictData.json"):
            raise Exception(f"The file predictData.json does not exist.")
        with open("predictData.json", 'r') as f:
            predict_data = json.load(f)
        df = pd.read_csv(sys.argv[1], sep=",")
        get_predictions(df, predict_data)
    except Exception as e:
        print(f"Exeption Error: {e}")
