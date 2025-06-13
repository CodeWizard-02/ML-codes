import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# load dataset
df = pd.read_csv("/home/akash/Datasets/homeprice.csv")
# print(df.head())

# check missing values
# print(df.isna().sum())

# create X and y
X = df['area'].values.reshape(-1, 1)
y = df['price']

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# regression model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred)

# model score
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Means Squeared Error: {mse}")
print(f"Means Absolute Error: {mae}")