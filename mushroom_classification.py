# -*- coding: utf-8 -*-
"""Mushroom Classification Using Machine Learning.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1irzIglYFckSQ9RETxERee7ijLQwhGLT0
"""

import pandas as pd

!pwd

# prompt: upload file

from google.colab import files
uploaded = files.upload()
for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))

# now you can use the uploaded file, e.g. with pandas:
# df = pd.read_csv(fn)
# print(df.head())

data = pd.read_csv('mushrooms.csv')

pd.set_option('display.max_columns',None)

"""### 1. Display Top 5 Rows of The Dataset"""

data.head()

#     Attribute Information: (classes: edible=e, poisonous=p)

#     cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s

#     cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s

#     cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y

#     bruises: bruises=t,no=f

#     odor: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s

#     gill-attachment: attached=a,descending=d,free=f,notched=n

#     gill-spacing: close=c,crowded=w,distant=d

#     gill-size: broad=b,narrow=n

#     gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y

#     stalk-shape: enlarging=e,tapering=t

#     stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?

#     stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s

#     stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s

#     stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y

#     stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y

#     veil-type: partial=p,universal=u

#     veil-color: brown=n,orange=o,white=w,yellow=y

#     ring-number: none=n,one=o,two=t

#     ring-type: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z

#     spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y

#     population: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y

#     habitat: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d

"""### 2. Check Last 5 Rows of The Dataset"""

data.tail()

"""### 3. Find Shape of Our Dataset (Number of Rows And Number of Columns)"""

data.shape

print("Number of Rows",data.shape[0])
print("Number of Columns",data.shape[1])

"""### 4. Get Information About Our Dataset Like Total Number Rows, Total Number of Columns, Datatypes of Each Column And Memory Requirement"""

data.info()

"""### 5. Check Null Values In The Dataset"""

data.isnull().sum()

"""### 6. Get Overall Statistics About The Dataset"""

data.describe()

"""### 7. Data Manipulation"""

data.head()

data.info()

data = data.astype('category')

data.dtypes

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for column in data.columns:
    data[column]=le.fit_transform(data[column])

data.head()

"""### 8. Store Feature Matrix In X and Response(Target) In Vector y"""

X = data.drop('class',axis=1)
y = data['class']

"""### 9. Applying PCA"""

from sklearn.decomposition import PCA

pca1 = PCA(n_components = 7)
pca_fit1 = pca1.fit_transform(X)

"""### 10. Splitting The Dataset Into The Training Set And Test Set"""

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(pca_fit1,y,test_size=0.20,
                                               random_state=42)

"""### 11. Import the models"""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

"""### 12. Model Training"""

lr = LogisticRegression()
lr.fit(X_train,y_train)

knn = KNeighborsClassifier()
knn.fit(X_train,y_train)

svc = SVC()
svc.fit(X_train,y_train)

dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)

rm = RandomForestClassifier()
rm.fit(X_train,y_train)

gb = GradientBoostingClassifier()
gb.fit(X_train,y_train)

"""### 13. Prediction on Test Data"""

y_pred1 = lr.predict(X_test)
y_pred2 = knn.predict(X_test)
y_pred3 = svc.predict(X_test)
y_pred4 = dt.predict(X_test)
y_pred5 = rm.predict(X_test)
y_pred6 = gb.predict(X_test)

"""### 14. Evaluating the Algorithm"""

from sklearn.metrics import accuracy_score

print("ACC LR",accuracy_score(y_test,y_pred1))
print("ACC KNN",accuracy_score(y_test,y_pred2))
print("ACC SVC",accuracy_score(y_test,y_pred3))
print("ACC DT",accuracy_score(y_test,y_pred4))
print("ACC RM",accuracy_score(y_test,y_pred5))
print("ACC GBC",accuracy_score(y_test,y_pred6))

final_data = pd.DataFrame({'Models':['LR','KNN','SVC','DT','RM','GBC'],
             'ACC': [accuracy_score(y_test,y_pred1)*100,
                    accuracy_score(y_test,y_pred2)*100,
                    accuracy_score(y_test,y_pred3)*100,
                    accuracy_score(y_test,y_pred4)*100,
                    accuracy_score(y_test,y_pred5)*100,
                    accuracy_score(y_test,y_pred6)*100]})

final_data

import seaborn as sns

sns.barplot(x=final_data['Models'], y=final_data['ACC'])
# Providing the 'x' and 'y' data as keyword arguments to the sns.barplot function

"""### Save The Model"""

rf_model = RandomForestClassifier()
rf_model.fit(pca_fit1,y)

import joblib

joblib.dump(rf_model,"Mushroom_prediction")

model = joblib.load('Mushroom_prediction')



p =model.predict(pca1.transform([[5,2,4,1,6,1,0,1,4,0,3,2,2,7,7,0,2,1,4,2,3,5]]))

if p[0]==1:
    print('Poisonous')
else:
    print('Edible')
