from describe import describe, pd, np
import matplotlib.pyplot as plt


def histogram(describ : describe):
    Houses = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
    df = describ._DataFrame
    numeric_columns = df.select_dtypes(include='number').columns
    numeric_columns = numeric_columns.drop('Index')
    numeric_df = df[numeric_columns].copy()
    Data = {column: pd.DataFrame(columns=Houses, dtype=float) for column in numeric_df.columns.values}
    numeric_df["Hogwarts House"] = df["Hogwarts House"]
    for _, row in numeric_df.iterrows():
        house = row["Hogwarts House"]
        row_data = row.drop('Hogwarts House')
        for column_name, column_data in row_data.items():
            Data[column_name].loc[Data[column_name][house].count(), house] = column_data

    NumOfFeatures = len(Data)
    NumOfCols = 4
    NumOfRows = (NumOfFeatures + NumOfCols - 1) // NumOfCols
    _, axes = plt.subplots(nrows=NumOfRows, ncols=NumOfCols, figsize=(20, 3 * NumOfRows), sharex=False, sharey=False)
    axes = axes.flatten()

    for ax, (feature, df) in zip(axes, Data.items()):
        df.plot(kind='hist', alpha=0.5, bins=30, ax=ax, legend=False)  # Set legend=False to avoid repetition
        ax.set_title(f'Homogeneity Scores for {feature}', fontsize=8)
        ax.legend(title='Houses', fontsize=8, loc='upper left')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    describ = describe()
    describ.calculate_data("datasets/dataset_train.csv")
    histogram(describ)
