import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot


######################   Get the Data  ########################################
dataframe = pd.read_csv("Train_v2.csv")
dataframe = dataframe.drop(["uniqueid"], axis=1) # drop unique_id (won't help analyzing data)


###################### Exploratory Data Analysis ##############################
dataframe.info() 
dataframe.head()
dataframe.shape
dataframe.columns
dataframe.index
dataframe.describe() 
dataframe["bank_account"].value_counts() # bank_account = target

# Missing or Null Data points
dataframe.isnull().sum()
dataframe.isna().sum()

# plotting target with countplot:
np.unique(dataframe["bank_account"], return_index=False, return_inverse=False, return_counts=True, axis=None)
sns.countplot(dataframe["bank_account"],label="Count")

# Boxplots:
# Boxplot of bank_account by location_type
trace0 = go.Box(
    y=dataframe.loc[dataframe['location_type'] == 'Rural']['bank_account'],
    name = 'With air conditioning',
    marker = dict(
        color = 'rgb(214, 12, 140)')
)
trace1 = go.Box(
    y=dataframe.loc[dataframe['location_type'] == 'Urban']['bank_account'],
    name = 'no air conditioning',
    marker = dict(
        color = 'rgb(0, 128, 128)',
    )
)
data = [trace0, trace1]
layout = go.Layout(
    title = "Boxplot of Sale Price by air conditioning"
)

fig = go.Figure(data=data,layout=layout)
iplot(fig)

# Dividing data into X_train and y_train
X = dataframe.drop(["bank_account"], axis=1)
Y = dataframe["bank_account"]
# LabelEncoding and OneHotEncoding for categorical data
labelencoder = LabelEncoder()
Y = labelencoder.fit_transform(Y)
X = pd.get_dummies(X)
# correlation matrix and heatmap can't give anything before encoding categorical
# features:
# correlation between features with heatmap
corrmat = X.corr()
corrmat
top_corr_features = corrmat.index
plt.figure(figsize=(30,30))
plt.savefig("output.png") # ? doesn't work
plt.show()
#plot heat map
g=sns.heatmap(X[top_corr_features].corr(),annot=True,cmap="RdYlGn")