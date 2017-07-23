
#import the necessary stuff
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 
import matplotlib
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#dataframes
titanic_df = pd.read_csv("train.csv") #training data frame
test_df    = pd.read_csv("test.csv")# test data frame 

#visualization of training dataframe . using spyder helps here .
titanic_df.head()
test_df.head()

#to priunt number of rows and columns in dataframes
print (titanic_df.shape,test_df.shape)

#here begins the feature eginnering .We try to excat Mr , Ms etc from every name
#https://stackoverflow.com/questions/45256435/extracting-word-from-a-sentence-using-split-in-python
def get_title(name):
    if '.' in name:
        name = name.split(',')[1]
        name = name.split('.')[0]
        name = name.strip()
        return name
    else:
       return 'Unknown'

def title_map(title):
    if title in ['Mr']:
        return 1
    elif title in ['Master']:
        return 3
    elif title in ['Ms','Mlle','Miss']:
        return 4
    elif title in ['Mme','Mrs']:
        return 5
    else:
        return 2
#we need add the new title column in the above dataframe and we will use the above two functions there 
titanic_df['title'] = titanic_df['Name'].apply(get_title).apply(title_map)   
test_df['title'] = test_df['Name'].apply(get_title).apply(title_map)
#https://chrisalbon.com/python/pandas_crosstabs.html       to learn about crosstab 
#visualization part of code 
title_xt = pd.crosstab(titanic_df['title'], titanic_df['Survived']) #look at varible explorer now to understand this 
title_xt_pct = title_xt.div(title_xt.sum(1).astype(float), axis=0)# to compare reducing the scale :)
title_xt_pct.plot(kind='bar', 
                  stacked=True, 
                  title='Survival Rate by title')
plt.xlabel('title')
plt.ylabel('Survival Rate') 
 #the figure shows clearly the importnace of name column  in dataset
titanic_df=titanic_df.drop(['PassengerId' ,'Name', 'Ticket'] , axis=1) #we dropped threee columns from training dataframe
test_df = test_df.drop(['Name', 'Ticket'], axis=1)  #drop two columns from test dataframe

titanic_df.count()
test_df.count()
#we  satrt working on embark column in dataframe
#https://medium.com/towards-data-science/the-dummys-guide-to-creating-dummy-variables-f21faddb1d40
titanic_df['Embarked']=titanic_df['Embarked'].fillna('S')#some values in titanic_df was missing so we filled it
embark = pd.get_dummies(titanic_df['Embarked'],drop_first=True)#for dummy varibale inplace -0f s and q
titanic_df.drop('Embarked',axis=1,inplace=True)#embart is out of our titanic_df data frame 
titanic_df = pd.concat([titanic_df,embark],axis=1)#we added embark onto our data time farme

embark = pd.get_dummies(test_df['Embarked'],drop_first=True)
test_df.drop('Embarked',axis=1,inplace=True)#from test dataframe we dropped the column embarked
test_df = pd.concat([test_df,embark],axis=1)#we concatanated embark onto our test dataframe

#we satrt working on sex column in datset
dummysex = pd.get_dummies(titanic_df['Sex'])#WE CREATED DUMMY VARIBLEES 0 ANND 1 FOR sex in our intila training dataframe
titanic_df.drop('Sex',axis=1,inplace=True)#from titanic we dropped sex column
titanic_df=pd.concat([titanic_df,dummysex],axis=1)#we added the dummysex column onto our training dataframe 
sex = pd.get_dummies(test_df['Sex'])
test_df.drop('Sex',axis=1,inplace=True)
test_df = pd.concat([test_df,sex],axis=1)

#we start working with the age column which is an important factor in saving life 
titanic_df['Age'].fillna(titanic_df['Age'].median() , inplace= True ) #there are some empty places in age column and we replace it with the median 
titanic_df['Age'] = titanic_df.groupby(['Pclass'])['Age'].transform(lambda x: x.fillna(x.mean()))
test_df['Age'] = test_df.groupby(['Pclass'])['Age'].transform(lambda x: x.fillna(x.mean()))#https://stackoverflow.com/questions/18689823/pandas-dataframe-replace-nan-values-with-average-of-columns
titanic_df['Age'] = titanic_df['Age'].astype(int)#converting to int datatype
test_df['Age']    = test_df['Age'].astype(int)#converting to int data type
titanic_df.loc[ titanic_df['Age'] <= 16, 'Age'] = 0 #.loc is used for aceesing from data frame and from here on we convert the age group into 0 1 2 3 4 depending upon conditiions .if age is less than 16 it belongs to first group called zero
titanic_df.loc[(titanic_df['Age'] > 16) & (titanic_df['Age'] <= 32), 'Age'] = 1
titanic_df.loc[(titanic_df['Age'] > 32) & (titanic_df['Age'] <= 48), 'Age'] = 2
titanic_df.loc[(titanic_df['Age'] > 48) & (titanic_df['Age'] <= 64), 'Age'] = 3
titanic_df.loc[(titanic_df['Age'] > 64), 'Age'] = 4
#we similarly do on testset
test_df.loc[ test_df['Age'] <= 16, 'Age'] = 0
test_df.loc[(test_df['Age'] > 16) & (test_df['Age'] <= 32), 'Age'] = 1
test_df.loc[(test_df['Age'] > 32) & (test_df['Age'] <= 48), 'Age'] = 2
test_df.loc[(test_df['Age'] > 48) & (test_df['Age'] <= 64), 'Age'] = 3
test_df.loc[(test_df['Age'] > 64), 'Age'] = 4
#this is a hack that has been used here as age and classs are quite related
titanic_df['age_class'] = titanic_df['Age'] * titanic_df['Pclass']
test_df['age_class'] = test_df['Age'] * test_df['Pclass']
#preprocessing fare part in train and test data frame
#we convert every price in 0,1,2,3 on the basis of conditions
test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)
titanic_df.loc[ titanic_df['Fare'] <= 7.91, 'Fare'] = 0
titanic_df.loc[(titanic_df['Fare'] > 7.91) & (titanic_df['Fare'] <= 14.454), 'Fare'] = 1
titanic_df.loc[(titanic_df['Fare'] > 14.454) & (titanic_df['Fare'] <= 31), 'Fare'] = 2
titanic_df.loc[ titanic_df['Fare'] > 31, 'Fare'] = 3
#same for test data frame 
test_df.loc[ test_df['Fare'] <= 7.91, 'Fare'] = 0
test_df.loc[(test_df['Fare'] > 7.91) & (test_df['Fare'] <= 14.454), 'Fare'] = 1
test_df.loc[(test_df['Fare'] > 14.454) & (test_df['Fare'] <= 31), 'Fare'] = 2
test_df.loc[test_df['Fare'] > 31, 'Fare'] = 3
# convert from float to int
titanic_df['Fare'] = titanic_df['Fare'].astype(int)
test_df['Fare']    = test_df['Fare'].astype(int)
#now we drop cabincolumn from training and test data
titanic_df.drop("Cabin",axis=1,inplace=True)
test_df.drop("Cabin",axis=1,inplace=True)
titanic_df.drop(['SibSp','Parch'], axis=1 ,inplace=True)
test_df.drop(['SibSp','Parch' ], axis=1,inplace=True)
#training and testing
X_train = titanic_df.drop("Survived",axis=1)#this is our training dataframe where we dropped survived column 
Y_train = titanic_df["Survived"]
X_test  = test_df.drop("PassengerId",axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)                
classifier.score(X_train, Y_train)
acc_random_forest = round(classifier.score(X_train, Y_train) * 100, 2)
acc_random_forest
print(acc_random_forest)

#fitting logistc regression to the training set 
from sklearn.linear_model import LogisticRegression
classifier1 = LogisticRegression(random_state=0)
classifier1.fit(X_train, Y_train)
# Predicting the Test set results
y_pred = classifier1.predict(X_test)
classifier1.score(X_train, Y_train)
acc_random_forest1 = round(classifier1.score(X_train, Y_train) * 100, 2)
print(acc_random_forest1)

# Fitting SVM classifier to the Training set
from sklearn.svm import SVC
classifier2 = SVC(kernel='linear', random_state=0)
classifier2.fit(X_train, Y_train)
# Predicting the Test set results
y_pred = classifier2.predict(X_test)   
classifier2.score(X_train, Y_train)
acc_random_forest2 = round(classifier2.score(X_train, Y_train) * 100, 2)
print(acc_random_forest2)
