from sklearn.datasets import fetch_california_housing
import pandas as pd

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['price'] = data.target

# Display the first few rows of the DataFrame
#print(df.head())
print("shape of the DataFrame:", df.shape)
print(df.describe)


    