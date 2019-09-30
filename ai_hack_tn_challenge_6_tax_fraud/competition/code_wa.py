import pandas as pd
import numpy as np
#Encoding categorical data values
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
​
​import os
print(os.getcwd())
​
""" Phase 1 : Data Exploration """
​
test_data_for_csv = pd.read_csv("competition/Test_v2_final.csv")
print(test_data_for_csv[(test_data_for_csv['marital_status']=='Married/Living together') & (test_data_for_csv['household_size']==1)])
for i in test_data_for_csv[(test_data_for_csv['marital_status']=='Married/Living together') & (test_data_for_csv['household_size']==1)]:
    print(i)
test_data_for_csv[(test_data_for_csv['marital_status']=='Married/Living together') & (test_data_for_csv['household_size']==1)].to_csv("false.csv")
test_data = pd.read_csv("competition/Test_v2.csv")
test_data[(test_data['marital_status']=='Married/Living together') & (test_data['household_size']==1)]
​
test_data.head()
test_data.columns
test_data = test_data.drop(['uniqueid', "year", "marital_status", "household_size", "relationship_with_head"], axis =1)
test_data.shape
​
​
​
test_data.shape # (10086, 11)
test_data.columns
test_data.info()
test_data.isnull().sum()
test_data.isna().sum()
​
​
train_data = pd.read_csv("/home/solixy-tessnim/Documents/IA/projects/competition/Train_v2.csv")
train_data.columns
train_data = train_data.drop(["uniqueid"], axis=1)
​
train_data.drop_duplicates(keep='first', inplace=True)
import os
train_data.to_csv(os.path.join("/home/solixy-tessnim/Documents/IA/projects/competition/train_droped_duplicated.csv"))
train_data = pd.read_csv("/home/solixy-tessnim/Documents/IA/projects/competition/train_droped_duplicated.csv", index_col=0)
train_data.shape
to_drop = train_data[(train_data['marital_status']=='Married/Living together') & (train_data['household_size']==1)].index
train_data = train_data.drop(to_drop)
​
train_data.columns
​
​
train_data.head()
train_data.shape # (23524, 13)
train_data.columns
train_data.info()
​
""" Missing or Null Data points """
train_data.isnull().sum()
train_data.isna().sum()
​
""" Separate the target variable and rest of the variables """
X = train_data.drop(["uniqueid", "bank_account", "year", "marital_status", "household_size", "relationship_with_head"], axis=1)
​
X = train_data.drop(["bank_account"], axis=1)
test_data.shape
X.shape # (23524, 11)
X.columns
X.info()
Y = train_data.bank_account
Y.shape
​
​
​
​
""" Visualisation of the target distribution """
train_data["bank_account"].value_counts()
np.unique(Y, return_index=False, return_inverse=False, return_counts=True, axis=None)
# (array(['No', 'Yes'], dtype=object), array([20212,  3312]))
sns.countplot(Y,label="Count")
plt.show()
​
""" Phase 2 : Categorical Data """
# 1 means have bank_account, 0 means does not have bank_account
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
Y
​
​
​
""" One hot encoding """
# One hot encode the variables
X = pd.get_dummies(X)
X.shape[1]
​
test_data = pd.get_dummies(test_data)
test_data.shape[1]
​
class MultiColumnLabelEncoder:
    
    def __init__(self, columns=None):
        self.columns = columns  # list of column to encode
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
​
​
​
labelencoder_X = MultiColumnLabelEncoder()
X = labelencoder_X.fit_transform(X)
X
​
​
test_df = labelencoder_X.fit_transform(test_data)
​
​
""" correlation between features with heatmap """
corrmat = X.corr()
corrmat
top_corr_features = corrmat.index
plt.figure(figsize=(15,15))
#plot heat map
g=sns.heatmap(X[top_corr_features].corr(),annot=True,cmap="RdYlGn")
​
​
""" split the data into train and test data """
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)
X_train.shape[1]
test_data.shape
""" apply StandardScaler """
​
​
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
​
test_df = sc.transform(test_df)
​
​
​
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
Y = sc_X.fit_transform(Y)
​
​
X_train.shape
X_test.shape
y_train.shape
y_test.shape
​
X.info()
​
from catboost import CatBoostClassifier
​
cat = CatBoostClassifier()
​
cat.fit(X_train, y_train)
y_pre_train = cat.predict(X_train)
accuracy_train = accuracy_score(y_train, y_pre_train)
​
y_pre_test = cat.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pre_test)
​
y_pred_final_set = cat.predict(test_data)
submission_df = pd.DataFrame({"uniqueid": test_data_for_csv["uniqueid"] + " x " + test_data_for_csv["country"], "bank_account": y_pred_final_set})
path = "bank/submission2"
import os
submission_df.to_csv(os.path.join(path, "SubmissionFile.csv"),index=False)
​
cat.plot_tree(1)
plt.show()
​
from xgboost import XGBClassifier
clf = XGBClassifier()
clf.fit(X_train,y_train)
#clf.fit(train_x, train_y, early_stopping_rounds=20, eval_set=[(test_x, test_y)])
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)
print(" xgboost: %.2f%%" %(acc_test*100))
​
​
​
​
y_pred_final_set = clf.predict(test_data)
submission_df = pd.DataFrame({"uniqueid": test_data_for_csv["uniqueid"] + " x " + test_data_for_csv["country"], "bank_account": y_pred_final_set})
path = "bank/submission2"
import os
submission_df.to_csv(os.path.join(path, "SubmissionFile.csv"),index=False)
​
​
​
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,y_pred_test) # Returns a ndarray
confusion_matrix
​
import matplotlib.pyplot as plt
import seaborn as sns
​
cm_df = pd.DataFrame(
    confusion_matrix,
    index = [idx for idx in ['yes', 'no']],
    columns = [col for col in ['yes', 'no']])
plt.figure(figsize = (10,7))
sns.heatmap(cm_df, annot=True)
​
​
​
​
from sklearn.naive_bayes import GaussianNB
# Initialize our classifier
gnb = GaussianNB()
​
# Train our classifier
model = gnb.fit(X_train, y_train)
​
# Make predictions
preds = gnb.predict(X_test)
print(preds)
print(y_test)
print('The accuracy using GaussianNB is :', accuracy_score(y_test, preds)) #0.8386822529224229
​
​
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
​
​
import os
path = "bank/submission2"
# Evaluate several ml models by training on training set and testing on testing set
def evaluate(X_train, X_test, y_train, y_test, test_data_for_csv):
   # Names of models
   model_name_list = ['Logistic Regression', 'Random Forest',
                      'Naive Bayes', 'Decision Tree', 'Nearest Neighbors',
                      'LDA', 'SVC', 'XGBoost', 'AdaBoost', 'QDA']
​
   # Instantiate the models
   model1 = LogisticRegression()
   model2 = RandomForestClassifier()
   model3 = GaussianNB()
   model4 = DecisionTreeClassifier()
   model5 = KNeighborsClassifier(n_neighbors=11)
   model6 = LinearDiscriminantAnalysis()
   model7 = SVC(kernel="linear", C=6)
   model8 = xgb.XGBClassifier(objective="binary:logistic")
   model9 = AdaBoostClassifier(n_estimators=100, random_state=0)
   model10 = QuadraticDiscriminantAnalysis()
​
   # Dataframe for results (on the test set)
   results_test = pd.DataFrame(columns=['mae_test', 'rmse_test', 'accuracy_score_test'], index=model_name_list)
   results_train = pd.DataFrame(columns=['mae_train', 'rmse_train', 'accuracy_score_train'], index=model_name_list)
​
   # Train and predict with each model
   for i, model in enumerate([model1, model2, model3, model4, model5, model6, model7, model8, model9, model10]):
       model.fit(X_train, y_train)
       y_pred_test = model.predict(X_test)
       y_pred_train = model.predict(X_train)
​
       predictions = model.predict(test_data)
       submission_df = pd.DataFrame(
           {"uniqueid": test_data_for_csv["uniqueid"] + " x " + test_data_for_csv["country"], "bank_account": predictions})
       submission_df.to_csv(os.path.join(path, str(model_name_list[i])+".csv"), index=False)
​
       # Metrics on test set
       mae_test = np.mean(abs(y_pred_test - y_test))
       rmse_test = np.sqrt(np.mean((y_pred_test - y_test) ** 2))
       test_accuracy_score = accuracy_score(y_pred_test, y_test)
​
       # Metrics on train set
       mae_train = np.mean(abs(y_pred_train - y_train))
       rmse_train = np.sqrt(np.mean((y_pred_train - y_train) ** 2))
       train_accuracy_score = accuracy_score(y_pred_train, y_train)
​
       # Insert results into the dataframe
       model_name = model_name_list[i]
       results_test.loc[model_name, :] = [mae_test, rmse_test, test_accuracy_score]
       results_train.loc[model_name, :] = [mae_train, rmse_train, train_accuracy_score]
​
​
   return results_test, results_train
​
results_test, results_train= evaluate(X_train, X_test, y_train, y_test, test_data_for_csv)
​
print("results on the test set :", results_test)
print("results on the train set :", results_train)
​
​
import matplotlib
from IPython.core.pylabtools import figsize
​
figsize(12, 8)
matplotlib.rcParams['font.size'] = 16
# Root mean squared error
ax =  plt.subplot(1, 2, 1)
results_test.sort_values('mae', ascending = True).plot.bar(y = 'mae', color = 'b', ax = ax)
plt.title('Model Mean Absolute Error'); plt.ylabel('MAE');
​
# Median absolute percentage error
ax = plt.subplot(1, 2, 2)
results_test.sort_values('rmse', ascending = True).plot.bar(y = 'rmse', color = 'r', ax = ax)
plt.title('Model Root Mean Squared Error'); plt.ylabel('RMSE');
​
plt.tight_layout()
​
​
​
submission_df = pd.DataFrame({"uniqueid": test_data["uniqueid"] + " x " + test_data["country"], "bank_account": predictions})
submission_df.to_csv("final_submession2.csv",index=False)
​
​
​
​
""" Keras neural network """
​
X_train.shape
from keras.models import Sequential
from keras.layers import Dense, Dropout
​
model = Sequential()
model.add(Dense(100, activation="relu", input_shape = (X_train.shape[1],))) # Hidden Layer 1 that receives the Input from the Input Layer
​
model.add(Dense(100, activation="relu")) # Hidden Layer 2
​
model.add(Dense(1, activation='sigmoid'))
​
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
​
model.fit(X_train, y_train, batch_size=50, nb_epoch=300)
model.summary() #Print model Summary
​
y_pred_train_set = model.predict(X_train)
y_pred_train_set = y_pred_train_set.round()
y_pred_train_set.shape
y_train.shape
​
​
acc_tr = accuracy_score(y_train, y_pred_train_set)
​
​
y_pred_test_set = model.predict(X_test)
y_pred_test_set = y_pred_test_set.round()
y_pred_test_set.shape
y_test.shape
​
​
acc_tes = accuracy_score(y_test, y_pred_test_set)
​
y_pred_final_set = model.predict(X_test)
y_pred_final_set = y_pred_final_set.round()
y_pred_final_set.shape
​
​
acc_tes = accuracy_score(y_test, y_pred_test_set)
​
y_pred_final_set = model.predict(test_data)
type(y_pred_final_set)
y_pred_final_set = y_pred_final_set.round()
y_pred_final_set = y_pred_final_set.astype(int)
y_pred_final_set = np.array([ elem for singleList in y_pred_final_set for elem in singleList])
​
​
submission_df = pd.DataFrame({"uniqueid": test_data_for_csv["uniqueid"] + " x " + test_data_for_csv["country"], "bank_account": y_pred_final_set})
path = "bank/submission2"
import os
submission_df.to_csv(os.path.join(path, "SubmissionFile.csv"),index=False)
