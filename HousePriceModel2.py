import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

# fetch the file
file_path = "/home/akash/Datasets/homeprice2.csv"
df = pd.read_csv(file_path)

# handle missing values
df["bedrooms"] = df["bedrooms"].fillna(df["bedrooms"].mean())
df["bedrooms"] = df["bedrooms"].apply(math.floor)

# create X and y
X = df.drop(columns = ["price"], axis = 1)
y = df["price"]

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# model build
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred)

# prediction
new_pred = model.predict([[1800, 3, 18]])
print(new_pred)

# model score
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")