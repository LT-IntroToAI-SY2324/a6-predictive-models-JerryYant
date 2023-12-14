import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("part4-classification/suv_data.csv")
data['Gender'].replace(['Male','Female'],[0,1],inplace=True)

x = data[["Age", "EstimatedSalary", "Gender"]].values
y = data["Purchased"].values

# Step 1: Print the values for x and y
print(x)
print(y)
# Step 2: Standardize the data using StandardScaler, 
scaler = StandardScaler().fit(x)
# Step 3: Transform the data
x = scaler.transform(x)
# Step 4: Split the data into training and testing data
xtrain, xtest, ytrain, ytest, = train_test_split(x, y, test_size= .2)
# Step 5: Fit the data

# Step 6: Create a LogsiticRegression object and fit the data
model = linear_model.LogisticRegression().fit(xtrain, ytrain)
model.fit(xtrain, ytrain)
# Step 7: Print the score to see the accuracy of the model
print("The accuracy is ", model.score(xtest, ytest))
print("-----------")
print("Test results")
print("")

# Step 8: Print out the actual ytest values and predicted y values
# based on the xtest data
print("Predicted y-value: " + str(model.predict(xtest)) + "Actual y-value: " + str(ytest))

newVar = [["34", "56000", "1"]]
newVar = scaler.transform(newVar)
prediction = model.predict(newVar)
print(prediction)