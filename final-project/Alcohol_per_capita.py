import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

data = pd.read_csv("final-project/alcohol_per_capita.csv")
x = data('TIME', 'LITRES/CAPITA')

x_std = StandardScaler().fit_transform(x)


