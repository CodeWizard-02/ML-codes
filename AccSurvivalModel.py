import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

# fetch dataset
file_path = "/home/akash/Datasets/accident.csv"
df = pd.read_csv(file_path)

# handle missing values
df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
df["Speed_of_Impact"] = df["Speed_of_Impact"].fillna(df["Speed_of_Impact"].mean())

# encode the Gender column
encoder = LabelEncoder()
df["Gender"] = encoder.fit_transform(df["Gender"])
df["Helmet_Used"] = encoder.fit_transform(df["Helmet_Used"])
df["Seatbelt_Used"] = encoder.fit_transform(df["Seatbelt_Used"])

# create X and y
X = df.drop(columns = ["Survived"], axis = 1)
y = df["Survived"]

# scale data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2)

# model build
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# model score
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))