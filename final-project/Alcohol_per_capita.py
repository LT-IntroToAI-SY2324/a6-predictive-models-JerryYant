
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

data=pd.read_csv("oecd-plots.ipynb")

x = data["Time"]
y = data["Litres/Capita"]

print(x)
print(y)