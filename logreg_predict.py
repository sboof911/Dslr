import numpy as np
import pandas as pd
import json, csv

def normalizedata(df : pd.DataFrame, means, stds):
    df = df.select_dtypes(include='number')
    df.drop('Index', axis=1, inplace=True)
    df.dropna(axis=1, inplace=True, how='all')
    NormalizedDF = df.copy()
    for col in df.columns:
        if col in means:
            NormalizedDF[col] = NormalizedDF[col].fillna(means[col])
            NormalizedDF[col] = (df[col] - means[col]) / stds[col]
    return NormalizedDF

def get_predictions(df: pd.DataFrame, predictData: dict):
    normalized_df = normalizedata(df, predictData["means"], predictData["stds"])
    features_values = normalized_df.values
    weights = predictData["weights"]
    predictions = []

    for _, weight in weights.items():
        weight_matrix = np.array(weight)
        y_pred = 1 / (1 + np.exp(-np.dot(features_values, weight_matrix)))
        predictions.append(y_pred)
    predictions = np.array(predictions)
    house_predictions = np.argmax(predictions, axis=0)
    HouseToInt = {'Ravenclaw': 0, 'Slytherin': 1, 'Gryffindor': 2, 'Hufflepuff': 3}
    IntToHouse = {v: k for k, v in HouseToInt.items()}
    return [IntToHouse[pred] for pred in house_predictions]

if __name__ == "__main__":
    import sys, os

    try:
        if len(sys.argv) != 2:
            raise Exception("Enter just the filepath as an argument!")
        if not sys.argv[1].endswith(".csv"):
            raise Exception(f"The file '{sys.argv[1]}' must be a CSV file.")
        if not os.path.exists(sys.argv[1]):
            raise Exception(f"The file '{sys.argv[1]}' does not exist.")
        if not os.path.exists("predictData.json"):
            raise Exception("The file predictData.json does not exist.")

        with open("predictData.json", 'r') as f:
            predict_data = json.load(f)

        df = pd.read_csv(sys.argv[1])
        predictions = get_predictions(df, predict_data)

        with open("houses.csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Index', 'Hogwarts House'])
            for idx, house in enumerate(predictions):
                writer.writerow([idx, house])

        print("Predictions saved to 'houses.csv'")

    except Exception as e:
        print(f"Exception Error: {e}")
