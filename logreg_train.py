from describe import describe, np
import json

class SortingHat(describe):
    def __init__(self, dataset_train_csvpath : str):
        super().__init__()
        self._ColumnsToDrop = ["Care of Magical Creatures", "Defense Against the Dark Arts"]
        self.calculate_data(dataset_train_csvpath)
        self.trainModel()

    def normalizeData(self, HouseToInt):
        self._SummaryDF.drop(self._ColumnsToDrop, axis=1, inplace=True)
        self._DataFrame.drop(self._ColumnsToDrop, axis=1, inplace=True)
        mean = self._SummaryDF.loc["Mean"]
        std  = self._SummaryDF.loc["Std"]
        self._NormalizedDF = self._DataFrame.copy()

        self._NormalizedDF["Hogwarts House"] = self._NormalizedDF["Hogwarts House"].map(HouseToInt)
        for col in self._DataFrame.columns:
            if col in mean.index:
                self._NormalizedDF[col] = (self._DataFrame[col] - mean[col]) / std[col]
                self._NormalizedDF[col] = self._NormalizedDF[col].fillna(0.0)
            elif col != "Hogwarts House":
                self._NormalizedDF.drop(col, axis=1, inplace=True)
        self.initialize_parameters(mean.index.shape[0])
        return self._NormalizedDF[self._NormalizedDF.columns.drop("Hogwarts House")].values, self._NormalizedDF["Hogwarts House"].values

    def initialize_parameters(self, featuresNum):
        self._Weight  = np.random.randn(featuresNum)

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
        return self.SigmoidFunction(np.dot(features_values, self._Weight))

    def evaluate_accuracy(self, y_true, y_pred):
        predictions = np.where(y_pred >= 0.5, 1, 0)
        return np.mean(y_true == predictions)

    def optimize(self, features_values : np.ndarray, TruetargetsValues : list, learning_rate : float, num_iterations : int, batch_size : float=0.3):
        featuresSize = features_values.shape[0]
        num_batches = int(featuresSize * batch_size)
        best_accuracy = 0
        best_Weight = self._Weight
        best_cost = 100
        for i in range(num_iterations):
            idx = np.random.permutation(featuresSize)
            feature_shuffled = features_values[idx]
            Truetarget_shuffled = TruetargetsValues[idx]
            feature_batch = feature_shuffled[:num_batches]
            Truetarget_batch = Truetarget_shuffled[:num_batches]
            self.gradient_descent(feature_batch, Truetarget_batch, learning_rate)
            cost = self.compute_cost(feature_batch, Truetarget_batch)

            if i % 100 == 0:
                y_pred_remaining = self.predict(features_values[num_batches:])
                accuracy = self.evaluate_accuracy(TruetargetsValues[num_batches:], y_pred_remaining)
                if best_accuracy < accuracy:
                    best_accuracy = accuracy
                    best_Weight = self._Weight
                    best_cost = cost
                print(f"Iteration {i}, cost: {cost}, Accuracy on remaining data: {accuracy * 100:.2f}%")

        print(f"Best cost: {best_cost}, and best Accuracy on all the data of the current House: {best_accuracy * 100:.2f}%")
        self._Weight = best_Weight
        print("-------------------------------------------")

    def trainModel(self):
        learning_rate = 0.1
        num_iterations = 500
        batch_size = 0.3  # % of data as batch
        HouseToInt = {'Ravenclaw': 0, 'Slytherin': 1, 'Gryffindor': 2, 'Hufflepuff': 3}
        features_values, classes_values = self.normalizeData(HouseToInt)

        weights = {}
        for house, _ in HouseToInt.items():
            print("House: ", house)
            y_one_vs_all = np.where(classes_values == HouseToInt[house], 1, 0)
            self.optimize(features_values, y_one_vs_all, learning_rate, num_iterations, batch_size)
            weights[house] = self._Weight.tolist()
            self._Weight = np.random.randn(self._Weight.shape[0])

        data = {
            "weights":weights,
            "means":self._SummaryDF.loc["Mean"].to_dict(),
            "stds":self._SummaryDF.loc["Std"].to_dict(),
            "ColumnsToDrop":self._ColumnsToDrop,
            "HouseToInt":HouseToInt
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