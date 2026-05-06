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


import pandas as pd

from sklearn.linear_model import LinearRegression


data = {
    "hours":[1,2,3,4,5],
    "result":[2,4,6,8,10]
}

df=pd.DataFrame(data)

X = df[["hours"]]
y= df["result"]

model = LinearRegression()
model.fit(X,y)

new_data = pd.DataFrame({
    "hours": [10]
})

prediction = model.predict(new_data)

print("prediction is", prediction)