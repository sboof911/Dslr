import matplotlib.pyplot as plt
import seaborn as sns
from describe import dslr

def pair_plot(df):
    numeric_df = df.select_dtypes(include='number')
    numeric_df = numeric_df.drop(['Index'], axis=1)
    numeric_df_with_house = numeric_df.join(df['Hogwarts House'])

    sns.pairplot(numeric_df_with_house, hue='Hogwarts House', diag_kind='hist')
    plt.show()
    
test = dslr("datasets/dataset_train.csv")
pair_plot(test.df)