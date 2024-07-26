from describe import dslr, np
import json

class SortingHat(dslr):
    def __init__(self, dataset_train_csvpath : str, bonus:bool = False):
        super().__init__(dataset_train_csvpath)
        self._bonus = bonus
        self._ColumnsToDrop = ["Care of Magical Creatures", "Defense Against the Dark Arts"]
        self._SummaryDF = self.describe().drop(['Index'], axis=1)
        self.trainModel()

    def normalizeData(self, HouseToInt):
        self._SummaryDF.drop(self._ColumnsToDrop, axis=1, inplace=True)
        self.df.drop(self._ColumnsToDrop, axis=1, inplace=True)
        mean = self._SummaryDF.loc["Mean"]
        std  = self._SummaryDF.loc["Std"]
        self._NormalizedDF = self.df.copy()
        if self.df["Hogwarts House"].isna().all():
            raise Exception("Predict Column is empty!")
        self._NormalizedDF["Hogwarts House"] = self._NormalizedDF["Hogwarts House"].map(HouseToInt)
        for col in self.df.columns:
            if col in mean.index:
                self._NormalizedDF[col] = (self.df[col] - mean[col]) / std[col]
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

    def gradient_descent(self, features_values, y_true):
        m = features_values.shape[0]
        if self._bonus:
            for i in range(m):
                x_i = features_values[i, :]
                y_i = y_true[i]
                y_pred = self.SigmoidFunction(np.dot(x_i, self._Weight))
                gradient = (y_pred - y_i) * x_i
                self._Weight -= self._learning_rate * gradient
        else:
            y_pred = self.SigmoidFunction(np.dot(features_values, self._Weight))
            gradient = np.dot(features_values.T, (y_pred - y_true)) / m
            self._Weight -= self._learning_rate * gradient

    def predict(self, features_values):
        return self.SigmoidFunction(np.dot(features_values, self._Weight))

    def evaluate_accuracy(self, y_true, y_pred):
        predictions = np.where(y_pred >= 0.5, 1, 0)
        return np.mean(y_true == predictions)

    def optimize(self, features_values : np.ndarray, TruetargetsValues : list):
        featuresSize = features_values.shape[0]
        num_batches = int(featuresSize * self._batch_size) if self._bonus else int(featuresSize)
        best_accuracy = 0
        best_Weight = self._Weight
        best_cost = 100
        for i in range(self._num_iterations):
            idx = np.random.permutation(featuresSize)
            feature_shuffled = features_values[idx]
            Truetarget_shuffled = TruetargetsValues[idx]
            feature_batch = feature_shuffled[:num_batches]
            Truetarget_batch = Truetarget_shuffled[:num_batches]
            self.gradient_descent(feature_batch, Truetarget_batch)
            cost = self.compute_cost(feature_batch, Truetarget_batch)
            if cost < best_cost and not self._bonus:
                    best_cost = cost
                    best_Weight = self._Weight
            if i % 100 == 0 and self._bonus:
                y_pred_remaining = self.predict(features_values[num_batches:])
                accuracy = self.evaluate_accuracy(TruetargetsValues[num_batches:], y_pred_remaining)
                if best_accuracy < accuracy:
                    best_accuracy = accuracy
                    best_Weight = self._Weight
                    best_cost = cost
                print(f"Iteration {i}, cost: {cost}, Accuracy on remaining data: {accuracy * 100:.2f}%")
        str_best_accuracy = f", and best Accuracy on all the data of the current House: {best_accuracy * 100:.2f}%"
        print(f"Best cost: {best_cost}{str_best_accuracy if self._bonus else ""}")
        self._Weight = best_Weight
        print("-------------------------------------------")

    def trainModel(self):
        self._learning_rate = 0.1
        self._num_iterations = 500
        self._batch_size = 0.3  # % of data as batch
        HouseToInt = {'Ravenclaw': 0, 'Slytherin': 1, 'Gryffindor': 2, 'Hufflepuff': 3}
        features_values, classes_values = self.normalizeData(HouseToInt)

        weights = {}
        for house, _ in HouseToInt.items():
            print("House: ", house)
            y_one_vs_all = np.where(classes_values == HouseToInt[house], 1, 0)
            self.optimize(features_values, y_one_vs_all)
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
    bonus = False
    try:
        if len(sys.argv) < 2 or len(sys.argv) > 3:
            raise Exception("\n  Args list:\n\tFirst_arg:DataSetTrain Path.\n\tSecond_arg:--bonus.")
        if len(sys.argv) == 3:
            if sys.argv[2] != "--bonus":
                raise Exception("Third argument must be --bonus")
            else:
                bonus = True
        if not sys.argv[1].endswith(".csv"):
            raise Exception(f"The file '{sys.argv[1]}' must be a csv file.")
        if not os.path.exists(sys.argv[1]):
            raise Exception(f"The file '{sys.argv[1]}' does not exist.")
        SortingHat = SortingHat(sys.argv[1], bonus)

    except Exception as e:
        print(f"Exeption Error: {e}")