import pandas as pd
import numpy as mnp

dataset=pd.read_csv('cryotherapy_dataset.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=100)

from sklearn.svm import SVC
model=SVC()
model.fit(x_train,y_train)
y_predict=model.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_predict)
print(confusion)

print(model.score(x_test,y_test))

import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))
plt.title('Classification')
plt.xlabel('x')
plt.ylabel('result_of_treatment')
plt.scatter(range(0,9),y_predict)