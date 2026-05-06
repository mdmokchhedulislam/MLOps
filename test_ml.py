from sklearn.linear_model import LinearRegression

X=[[10], [20], 30]
y=[100,200,300]


model = LinearRegression()

model.fit(X,y)

result=model.predict([[5]])

print("prediction", result)