# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
```
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.
```
# FEATURE SCALING:
```
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).
```
# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```py
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("/content/income.csv",na_values=[ " ?"])
data
```
<img width="1529" height="470" alt="image" src="https://github.com/user-attachments/assets/6b16e469-641d-4790-8df4-957965730b63" />

```py
data.isnull().sum()
```
<img width="199" height="547" alt="image" src="https://github.com/user-attachments/assets/b02cc005-d8e3-42c5-a92c-9364121fc6f4" />

```py
missing=data[data.isnull().any(axis=1)]
missing
```
<img width="1485" height="458" alt="image" src="https://github.com/user-attachments/assets/7ab8551b-3a8c-47fb-82a5-80c76390808f" />

```py
data2=data.dropna(axis=0)
data2
```
<img width="1527" height="463" alt="image" src="https://github.com/user-attachments/assets/47ffbe88-24a4-4254-83f3-3e81b3832037" />

```py
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
<img width="408" height="241" alt="image" src="https://github.com/user-attachments/assets/6cf62d89-3002-4d34-89b2-91a7faa0f9d9" />

```py
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
<img width="363" height="460" alt="image" src="https://github.com/user-attachments/assets/9621b75d-485d-4d2d-806a-7bcc20f7ca42" />

```py
data2
```
<img width="1394" height="465" alt="image" src="https://github.com/user-attachments/assets/16c09014-dda0-466e-a938-a8a622a9ed0d" />

```py
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
<img width="1686" height="528" alt="image" src="https://github.com/user-attachments/assets/eb25465e-4304-482a-bcc8-af7ef15aaf84" />

```py
columns_list=list(new_data.columns)
print(columns_list)
```
<img width="1667" height="42" alt="image" src="https://github.com/user-attachments/assets/b23c3333-af31-4107-8585-f540d19fda4e" />

```py
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
<img width="1656" height="44" alt="image" src="https://github.com/user-attachments/assets/6bbcd496-c879-4532-9871-1b7a92894cd2" />

```py
y=new_data['SalStat'].values
print(y)
```
<img width="187" height="38" alt="image" src="https://github.com/user-attachments/assets/77dcce98-779f-4069-a671-fe732dc01ac0" />

```py
x=new_data[features].values
print(x)
```
<img width="442" height="167" alt="image" src="https://github.com/user-attachments/assets/5abdfb9e-e837-4725-83d9-5e2563f640a9" />

```py
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
<img width="308" height="79" alt="image" src="https://github.com/user-attachments/assets/7a7f6d6f-1b44-4f04-a1db-c758aa3e921f" />

```py
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
<img width="154" height="57" alt="image" src="https://github.com/user-attachments/assets/ee450029-5c4a-4f4d-878a-7e5d68e7c33f" />

```py
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
<img width="191" height="28" alt="image" src="https://github.com/user-attachments/assets/bce6e3f6-b5ac-4c4e-a45e-69427ec1c6b6" />

```py
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
<img width="294" height="33" alt="image" src="https://github.com/user-attachments/assets/de8d8df4-598e-409d-953a-76e44fc678ac" />

```py
data.shape
```
<img width="134" height="34" alt="image" src="https://github.com/user-attachments/assets/600debd6-b3ba-43c1-bb33-a6fca36a69e2" />

```py
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
'Feature1': [1,2,3,4,5],
'Feature2': ['A','B','C','A','B'],
'Feature3': [0,1,1,0,1],
'Target' : [0,1,1,0,1]
}
df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
<img width="351" height="50" alt="image" src="https://github.com/user-attachments/assets/0e57aebf-5078-43f0-b88e-79edc06c85c7" />

```py
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
<img width="546" height="254" alt="image" src="https://github.com/user-attachments/assets/fdffb462-5e24-408b-82e1-ec386cfe92b5" />

```py
tips.time.unique()
```
<img width="448" height="56" alt="image" src="https://github.com/user-attachments/assets/7e22d348-8ef4-4aac-bc21-9c302ddd226c" />


```py
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```

<img width="228" height="95" alt="image" src="https://github.com/user-attachments/assets/70d1e944-7873-43a6-b44a-fe0dd7417fee" />


```py
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```

<img width="418" height="52" alt="image" src="https://github.com/user-attachments/assets/9b258b24-c3bb-49c4-ad30-ab3eed8252e5" />


# RESULT:
Thus the program to read the given data and perform Feature Scaling and Feature Selection process and
save the data to a file is been executed.
