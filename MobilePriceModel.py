import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# fetch dataset
df = pd.read_csv('/home/akash/Datasets/mobilePrice.csv')
print(df.head())