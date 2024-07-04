import pandas as pd
import numpy as np

class describe:
    _Dataframe : pd.DataFrame
    def __init__(self):
        self._Dataframe = pd.DataFrame()

    def calculate_percentile(self, array, percentile):
        index = (len(array) - 1) * percentile / 100
        lower = int(index)
        upper = lower + 1
        weight = index - lower
        if upper >= len(array):
            return array[lower]
        return array[lower] * (1 - weight) + array[upper] * weight

    def calculate_data(self, dataset_train_csvpath : str):
        df = pd.read_csv(dataset_train_csvpath, sep=",")
        df = df.select_dtypes(include='number')
        df.drop('Index', axis=1, inplace=True) # axis = 1 for columns and axis = 0 for rows
        df.dropna(axis=1, inplace=True, how='all')
        columnsData = {}
        for column_name, column_data in df.items():
            values = column_data.dropna().values
            values = np.sort(values)
            data = {}
            data["Count"] = len(values)
            data["Mean"] = np.sum(values) / len(values)
            data["Std"] = np.sqrt(sum((value - data["Mean"]) ** 2 for value in values) / len(values))
            data["Min"] = values[0]
            data["25%"] = self.calculate_percentile(values, 25)
            data["50%"] = self.calculate_percentile(values, 50)
            data["75%"] = self.calculate_percentile(values, 75)
            data["Max"] = values[-1]
            columnsData[column_name] = data
        self._Dataframe = pd.DataFrame(columnsData)

def Display_Data(dataset_train_csvpath):
    db = describe()
    db.calculate_data(dataset_train_csvpath)
    print(db._Dataframe)

if __name__ == "__main__":
    import sys, os

    try:
        if len(sys.argv) != 2:
            raise Exception("Enter just the filepath as an arguments!")
        if not sys.argv[1].endswith(".csv"):
            raise Exception(f"The file '{sys.argv[1]}' must be a csv file.")
        if not os.path.exists(sys.argv[1]):
            raise Exception(f"The file '{sys.argv[1]}' does not exist.")
        Display_Data(sys.argv[1])
    except Exception as e:
        print(e)