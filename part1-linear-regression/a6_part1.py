import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv("part1-linear-regression/blood_pressure_data.csv")
x = data["Age"].values
y = data["Blood Pressure"].values

# Use reshape to turn the x values into 2D arrays:
x = x.reshape(-1,1)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size= .2)

xtrain = xtrain.reshape(-1, 1)
# Create the model
model =LinearRegression().fit(x, y)

# Find the coefficient, bias, and r squared values. 
# Each should be a float and rounded to two decimal places. 
coef = round(float(model.coef_), 2)
intercept = round(float(model.intercept_), 2)
r_squared = model.score(x, y)
print(coef, intercept, r_squared)

# Print out the linear equation and r squared value
print(f"Model's Linear Equation: y = {coef}x + {intercept}")
print(f"R Squared value: {r_squared}")

# Predict the the blood pressure of someone who is 43 years old.
x_predict = 43
prediction = model.predict([[x_predict]])
# Print out the prediction
print(f"Prediction when x is {x_predict}: {prediction}")
# Create the model in matplotlib and include the line of best fit
plt.figure(figsize=(6,4))
plt.scatter(x,y)
plt.xlabel("Age")
plt.ylabel("Blood Pressure")
plt.title("Blood Pressure by Age")

plt.show