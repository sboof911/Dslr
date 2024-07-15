from describe import describe, np, pd
import json

class SortingHat(describe):
    def __init__(self, dataset_train_csvpath : str):
        super().__init__()
        self.calculate_data(dataset_train_csvpath)
        self.trainModel()

    def normalizeData(self, HouseToInt):
        mean = self._SummaryDF.loc["Mean"]
        std  = self._SummaryDF.loc["Std"]
        self._NormalizedDF = self._DataFrame.copy()

        self._NormalizedDF["Hogwarts House"] = self._NormalizedDF["Hogwarts House"].map(HouseToInt)
        for col in self._DataFrame.columns:
            if col in mean.index:
                self._DataFrame[col] = self._DataFrame[col].fillna(mean[col])
                self._NormalizedDF[col] = (self._DataFrame[col] - mean[col]) / std[col]
            elif col != "Hogwarts House":
                self._NormalizedDF.drop(col, axis=1, inplace=True)
        self.initialize_parameters(mean.index.shape[0], len(HouseToInt))
        return self._NormalizedDF[self._NormalizedDF.columns.drop("Hogwarts House")].values, self._NormalizedDF["Hogwarts House"].values

    def initialize_parameters(self, featuresNum, HousesNum):
        self._Weight  = np.random.randn(featuresNum, HousesNum)

    def SigmoidFunction(self, y):
        return 1 / (1 + np.exp(-y))

    def compute_cost(self, features_values, y_true):
        y_pred = self.SigmoidFunction(np.dot(features_values, self._Weight))
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        cost = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return cost

    def gradient_descent(self, features_values, y_true, learning_rate):
        m = features_values.shape[0]
        y_pred = self.SigmoidFunction(np.dot(features_values, self._Weight))
        gradient = np.dot(features_values.T, (y_pred - y_true)) / m
        self._Weight -= learning_rate * gradient

    def predict(self, features_values):
        y_pred = self.SigmoidFunction(np.dot(features_values, self._Weight))
        return np.argmax(y_pred, axis=1)

    def evaluate_accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

    def optimize(self, features_values : np.ndarray, TruetargetsValues : list, learning_rate : float, num_iterations : int, batch_size : float=0.3):
        featuresSize = features_values.shape[0]
        num_batches = int(featuresSize * batch_size)
        for i in range(num_iterations):
            idx = np.random.permutation(featuresSize)
            feature_shuffled = features_values[idx]
            Truetarget_shuffled = TruetargetsValues[idx]
            feature_batch = feature_shuffled[:num_batches]
            Truetarget_batch = Truetarget_shuffled[:num_batches]
            self.gradient_descent(feature_batch, Truetarget_batch, learning_rate)
            cost = self.compute_cost(feature_batch, Truetarget_batch)
            self._costs.append(cost)

            if i % 100 == 0:
                y_pred_remaining = self.predict(features_values[num_batches:])
                accuracy = self.evaluate_accuracy(np.argmax(TruetargetsValues[num_batches:], axis=1), y_pred_remaining)
                print(f"Iteration {i}, cost: {cost}, Accuracy on remaining data: {accuracy * 100:.2f}%")

    def trainModel(self):
        learning_rate = 1
        num_iterations = 1000
        batch_size = 0.3  # 30% of data as batch
        self._costs = []
        HouseToInt = {'Ravenclaw': 0, 'Slytherin': 1, 'Gryffindor': 2, 'Hufflepuff': 3}
        features_values, classes_values = self.normalizeData(HouseToInt)
        weights = {}
        for house, cls in HouseToInt.items():
            y_one_vs_all = np.zeros((classes_values.shape[0], len(HouseToInt)), dtype=int)
            for key, value in enumerate(classes_values):
                y_one_vs_all[key][value] = 1
            self.optimize(features_values, y_one_vs_all, learning_rate, num_iterations, batch_size)
            weights[house] = self._Weight.tolist()

        data = {
            "weights":weights,
            "means":self._SummaryDF.loc["Mean"].to_dict(),
            "stds":self._SummaryDF.loc["Std"].to_dict()
        }
        with open("predictData.json", 'w') as f:
            json.dump(data, f, indent=4)

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