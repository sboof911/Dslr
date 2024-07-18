from describe import describe, pd, np


def histogram(describ : describe):
    df = describ._DataFrame
    numeric_columns = df.select_dtypes(include='number').columns
    numeric_columns = numeric_columns.drop('Index')
    numeric_df = df[numeric_columns].copy()
    Data = {'Ravenclaw': pd.DataFrame(columns=numeric_df.columns.values),
            'Slytherin': pd.DataFrame(columns=numeric_df.columns.values),
            'Gryffindor' : pd.DataFrame(columns=numeric_df.columns.values),
            'Hufflepuff' : pd.DataFrame(columns=numeric_df.columns.values)}
    numeric_df["Hogwarts House"] = df["Hogwarts House"]
    for _, row in numeric_df.iterrows():
        house = row["Hogwarts House"]
        row_data = row.drop('Hogwarts House')
        Data[house] = Data[house] = pd.concat([Data[house], pd.DataFrame([row_data])], ignore_index=True)
    varianceData = {'Ravenclaw': pd.DataFrame(columns=Data['Ravenclaw'].columns.values),
        'Slytherin': pd.DataFrame(columns=Data['Slytherin'].columns.values),
        'Gryffindor' : pd.DataFrame(columns=Data['Gryffindor'].columns.values),
        'Hufflepuff' : pd.DataFrame(columns=Data['Hufflepuff'].columns.values)}
    for HouseName, DataFrame in Data.items():
        for column_name, column_data in DataFrame.items():
            mean = np.sum(column_data) / len(column_data)
            var = np.sum((column_data - mean) ** 2) / len(column_data)
            varianceData[HouseName].at[0, column_name] = var
    homoginity = {}
    
    print(varianceData['Ravenclaw'])

if __name__ == "__main__":
    describ = describe()
    describ.calculate_data("datasets/dataset_train.csv")
    histogram(describ)
