from describe import describe, np, pd

class SortingHat(describe):
    def __init__(self, dataset_train_csvpath : str):
        super().__init__()
        self.calculate_data(dataset_train_csvpath)
        self.trainModel()

    def normalizeData(self):
        mean = self._SummaryDF.loc["Mean"]
        std  = self._SummaryDF.loc["Std"]
        self._NormalizedDF = self._DataFrame.copy()
        HouseToInt = {'Ravenclaw': 0, 'Slytherin': 1, 'Gryffindor': 2, 'Hufflepuff': 3}
        self._NormalizedDF["Hogwarts House"] = self._NormalizedDF["Hogwarts House"].map(HouseToInt)
        for col in self._DataFrame.columns:
            if col in mean.index:
                self._NormalizedDF[col] = (self._DataFrame[col] - mean[col]) / std[col]
            elif not col == "Hogwarts House":
                self._NormalizedDF.drop(col, axis=1, inplace=True)
        self.initialize_parameters(mean.index.shape[0], len(HouseToInt))
        return self._NormalizedDF[self._NormalizedDF.columns.drop("Hogwarts House")].values, self._NormalizedDF["Hogwarts House"].values

    def initialize_parameters(self, featuresNum, HousesNum):
        self._Weight  = np.random.randn(featuresNum, HousesNum)
        self._Bias = np.zeros((1, HousesNum))

    def SigmoidFunction(self, y):
        return 1 / (1 + np.exp(-y))

    def optimize(self, features_values : np.ndarray, TruetargetsValues : list, learning_rate : float, num_iterations : int, batch_size : float=0.3):
        featuresSize = features_values.shape[0]
        num_batches = int(featuresSize * batch_size)
        for i in range(num_iterations):
            idx = np.random.permutation(featuresSize)
            feature_shuffled = features_values[idx]
            Truetarget_shuffled = TruetargetsValues[idx]

    def trainModel(self):
        learning_rates = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
        best_learning_rate = 100
        num_iterations = 1000
        min_cost = float('inf')
        tolerance = 1e-8
        batch_size = 0.3  # 30% of data as batch
        self._costs = []
        features_values, target_values = self.normalizeData()

        for learning_rate in learning_rates:
            self.optimize(features_values, target_values, learning_rate, num_iterations, batch_size)

if __name__ == "__main__":
    import sys, os

    try:
        if len(sys.argv) != 2:
            raise Exception("Enter just the filepath as an arguments!")
        if not sys.argv[1].endswith(".csv"):
            raise Exception(f"The file '{sys.argv[1]}' must be a csv file.")
        if not os.path.exists(sys.argv[1]):
            raise Exception(f"The file '{sys.argv[1]}' does not exist.")
        SortingHat = SortingHat(sys.argv[1])

    except Exception as e:
        print(f"Exeption Error: {e}")