import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import *
from xgboost import plot_importance
from sklearn.feature_extraction import FeatureHasher
#***********
test_data = pd.read_csv("Test_v2.csv")
del test_data["uniqueid"]
del test_data["relationship_with_head"]
del test_data["year"]
del test_data["marital_status"]
del test_data["household_size"]
test_data.isnull().sum()
test_data.isna().sum()
test_data = pd.get_dummies(test_data)
print(test_data.head())
train_data = pd.read_csv("Train_v2.csv")
Y = train_data.bank_account
del train_data["uniqueid"]
del train_data ["bank_account"]
del train_data["relationship_with_head"]
del train_data["year"]
del train_data ["marital_status"]
del train_data ["household_size"]
X = train_data
X = pd.get_dummies(X)
corr = X.corr()
print(corr)
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5,shuffle=True,random_state=0)
from xgboost import XGBClassifier
clf = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0.1,
             learning_rate=0.120214, max_delta_step=0, max_depth=3,
             min_child_weight=2, missing=None, n_estimators=100, n_jobs=1,
             nthread=None, objective='binary:logistic', random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=None, subsample=1, verbosity=1)
from sklearn.ensemble import ExtraTreesClassifier
clf.fit(X_train,y_train)
print(clf.feature_importances_)
pyplot.bar(range(len(clf.feature_importances_)), clf.feature_importances_)
plot_importance(clf)
pyplot.show()
#clf.fit(train_x, train_y, early_stopping_rounds=20, eval_set=[(test_x, test_y)])
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)
print(" xgboost: %.8f%%" %(acc_test*100))
test_data_for_csv = pd.read_csv("Test_v2.csv")
#test_data=pd.drop["uniqueid", "bank_account"], axis=1
y_pred_final_set = clf.predict(test_data)
submission_df = pd.DataFrame({"uniqueid": test_data_for_csv["uniqueid"] + " x " + test_data_for_csv["country"], "bank_account": y_pred_final_set})
import os
submission_df.to_csv("alxxx1.csv",index=False)