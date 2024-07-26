import matplotlib.pyplot as plt
import seaborn as sns
from describe import dslr

def histogram(df):
    courses = df.drop(['Index', 'Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand'], axis=1)
    courses = courses.dropna()
    
    courses_with_houses = courses.join(df['Hogwarts House'])
    
    plt.figure(figsize=(20, 16))

    for i, course in enumerate(courses.columns, 1):
        plt.subplot(5, 3, i)  # 5 rows and 3 columns
        sns.histplot(data=courses_with_houses, x=course, hue='Hogwarts House', element="step", bins=30)
        plt.title(course)
    plt.tight_layout()
    plt.show()

test = dslr("datasets/dataset_train.csv")
histogram(test.df)