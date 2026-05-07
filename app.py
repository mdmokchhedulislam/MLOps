####### simple practice with linearRegression

# from sklearn.linear_model import LinearRegression

# X=[[10], [20], 30]
# y=[100,200,300]


# model = LinearRegression()

# model.fit(X,y)

# result=model.predict([[5]])

# print("prediction", result)



# Data handle with pandas

# import pandas as pd

# #create data

# data={
#     "hours": [1,2,3,4,5],
#     "result":[2,4,6,8,10]
# }

# df=pd.DataFrame(data)
# print(df["hours"])



# simple project with pandas


# import pandas as pd

# from sklearn.linear_model import LinearRegression


# data = {
#     "hours":[1,2,3,4,5],
#     "result":[2,4,6,8,10]
# }

# df=pd.DataFrame(data)

# X = df[["hours"]]
# y= df["result"]

# model = LinearRegression()
# model.fit(X,y)

# new_data = pd.DataFrame({
#     "hours": [10]
# })

# prediction = model.predict(new_data)

# print("prediction is", prediction)





#simple project read data from csv file

# import pandas as pd
# from sklearn.linear_model import LinearRegression

# df = pd.read_csv("data.csv")

# # print("csv data is", df)

# X = df[["hours"]]
# y= df['result']

# model = LinearRegression()
# model.fit(X,y)

# new_data=pd.DataFrame({
#     "hours": [12]
# })

# prediction = model.predict(new_data)

# print("prediction is", prediction)


# model save project 

# import pandas as pd
# import joblib

# from sklearn.linear_model import LinearRegression

# df = pd.read_csv("data.csv")

# X=df[["hours"]]
# y=df["result"]

# model= LinearRegression()

# model.fit(X,y)

# joblib.dump(model, "model.pkl")

# print("model saved !")


# load model 

# import joblib
# import pandas as pd

# model = joblib.load("model.pkl")

# # print("model pkl is", model)


# new_data = pd.DataFrame({
#     "hours": [15]
# })

# prediction = model.predict(new_data)

# print("Prediction:", prediction)




# model serving simple project 

# from fastapi import FastAPI
# import joblib
# import pandas as pd

# app = FastAPI()

# model = joblib.load("model.pkl")

# @app.get("/")
# def home():
#     return {"message": "ml api is running"}

# @app.get("/predict")
# def predict(hours: int):
#     data = pd.DataFrame({"hours": [hours]})
#     prediction = model.predict(data)
#     return {"prediction": prediction.tolist()}


# mlflow
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# MLflow DB location
mlflow.set_tracking_uri("sqlite:///C:/mlflow/mlflow.db")

# experiment name
mlflow.set_experiment("wine-classification")

# dataset
wine = load_wine()

# print("wine data is", wine)

X = wine.data
y = wine.target

# split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.10,
    random_state=42
)

# params
max_depth = 5
n_estimators = 10

# start run
with mlflow.start_run():

    rf = RandomForestClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        random_state=42
    )

    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

#     # logging
    mlflow.log_metric("accuracy", accuracy)

    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)

#     # save model
    mlflow.sklearn.log_model(rf, "random_forest_model")
    print("Accuracy:", accuracy)