import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# fetch dataset
file_path = "/home/akash/Datasets/income.csv"
df = pd.read_csv(file_path)

# scaling
scaler = MinMaxScaler()
df["Income($)"] = scaler.fit_transform(df[["Income($)"]])
df["Age"] = scaler.fit_transform(df[["Age"]])

# clustering
km = KMeans(n_clusters = 3)
y_pred = km.fit_predict(df[["Age", "Income($)"]])
df["Cluster"] = y_pred
print(km.cluster_centers_)