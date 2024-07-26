import numpy as np
import pandas as pd

class dslr:
    def __init__(self, path):
        self.df = pd.read_csv(path)

    def describe(self):
        df = self.df.select_dtypes(include='number')
        df = df.dropna(axis=1, how='all')
        results = {}

        for column_name, column_data in df.items():
            values = column_data.dropna().values
            numeric_values = pd.to_numeric(values, errors='coerce')
            sorted_values = np.sort(numeric_values)
            count = len(sorted_values)
            
            mean = np.sum(sorted_values) / count
            std = np.sqrt(np.sum((sorted_values - mean) ** 2) / (count - 1))
            min_val = sorted_values[0]
            
            # Calculating quantiles without using np.percentile
            def quantile(sorted_vals, q):
                index = (len(sorted_vals) - 1) * q
                lower_index = int(np.floor(index))
                upper_index = int(np.ceil(index))
                if lower_index == upper_index:
                    return sorted_vals[lower_index]
                else:
                    interp = (sorted_vals[upper_index] - sorted_vals[lower_index]) * (index - lower_index)
                    return sorted_vals[lower_index] + interp

            q25 = quantile(sorted_values, 0.25)
            q50 = quantile(sorted_values, 0.5)
            q75 = quantile(sorted_values, 0.75)
            max_val = sorted_values[-1]
            range_val = max_val - min_val
            variance = std ** 2
        
            results[column_name] = {
                "Count": count,
                "Mean": mean,
                "Std": std,
                "Min": min_val,
                "25%": q25,
                "50%": q50,
                "75%": q75,
                "Max": max_val,
                "Variance": variance,
                "Range": range_val,
            }

        return pd.DataFrame(results)

if __name__ == "__main__":
    test = dslr("datasets/dataset_train.csv")
    print(test.describe())