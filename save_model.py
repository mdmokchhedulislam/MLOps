# model save project 

import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression

df = pd.read_csv("data.csv")

X=df[["hours"]]
y=df["result"]

model= LinearRegression()

model.fit(X,y)

joblib.dump(model, "model.pkl")

print("model saved !")