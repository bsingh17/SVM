import numpy as np
import pandas as pd
import seaborn as sns
dataset=pd.read_csv('pima_native_american_diabetes_dataset.csv')

x=pd.DataFrame(dataset.iloc[:,:-1].values)
y=pd.DataFrame(dataset.iloc[:,-1].values)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(x)
x=scaler.transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=0)

from sklearn.svm import SVC
model=SVC(kernel='linear')
model.fit(x_train,y_train)

y_predict=model.predict(x_test)
y_predict=pd.DataFrame(y_predict)
dataset['predicted']=y_predict

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_predict)

print(confusion)

print(model.score(x_test,y_test))

sns.regplot(x='age',y='predicted',data=dataset,logistic=True)