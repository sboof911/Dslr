import os
import pandas as pd
import numpy as np

class describe:
    _Dataframe : pd.DataFrame
    def __init__(self):
        self._Dataframe = None

    def count(self, df : pd.DataFrame):
        new_row_data = {}

        self._Dataframe.loc['Count'] = new_row_data

    def calculate_data(self, dataset_train_csvpath : str):

        if not dataset_train_csvpath.endswith(".csv"):
            raise Exception(f"The file '{dataset_train_csvpath}' must be a csv file.")
        if not os.path.exists(dataset_train_csvpath):
            raise Exception(f"The file '{dataset_train_csvpath}' does not exist.")

        df = pd.read_csv(dataset_train_csvpath, sep=",")
        df = df.select_dtypes(include='number')
        df.drop('Index', axis=1, inplace=True) # axis = 1 for columns and axis = 0 for rows
        df.dropna(axis=1, inplace=True, how='all')
        if self._Dataframe == None:
            self._Dataframe = pd.DataFrame(columns=df.columns)
        for column_name, column_data in df.items():
            values = column_data.dropna().values
            # sorted_values = 
        print(self._Dataframe)

def Display_Data(dataset_train_csvpath):
    db = describe()
    db.calculate_data(dataset_train_csvpath)

if __name__ == "__main__":
    import sys

    try:
        if len(sys.argv) != 2:
            raise Exception("Enter just the filepath as an arguments!")
        Display_Data(sys.argv[1])
    except Exception as e:
        print(e)