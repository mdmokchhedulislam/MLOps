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

from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("model.pkl")

@app.get("/")
def home():
    return {"message": "ml api is running"}

@app.get("/predict")
def predict(hours: int):
    data = pd.DataFrame({"hours": [hours]})
    prediction = model.predict(data)
    return {"prediction": prediction.tolist()}