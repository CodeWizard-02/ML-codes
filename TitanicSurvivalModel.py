import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# fetch dataset
file_path = "/home/akash/Datasets/titanic.csv"
df = pd.read_csv(file_path)

# remove unnecessary columns
df = df.drop(columns = ["Fare", "SibSp", "Parch", "Ticket", "Embarked", "Cabin", "Name"], axis = 1)

# handle missing values
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Age"] = df["Age"].apply(math.floor)

# Label encoding on Sex column
encoder = LabelEncoder()
df["Sex"] = encoder.fit_transform(df["Sex"])

# Create X and y
X = df.drop(columns = ["Survived"], axis = 1)
y = df["Survived"]

# scale X
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2)

# build model
model = RandomForestClassifier(n_estimators = 40)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred)

# model score
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))