import numpy as np
import pandas as pd

dataset=pd.read_csv('tic_tac_toe_dataset.csv')

from sklearn.preprocessing import LabelEncoder
lbl_topleft=LabelEncoder()
dataset['top_left_square']=lbl_topleft.fit_transform(dataset['top_left_square'])
lbl_topmiddle=LabelEncoder()
dataset['top_middle_square']=lbl_topmiddle.fit_transform(dataset['top_middle_square'])
lbl_topright=LabelEncoder()
dataset['top_right_square']=lbl_topright.fit_transform(dataset['top_right_square'])
lbl_middleleft=LabelEncoder()
dataset['middle_left_square']=lbl_middleleft.fit_transform(dataset['middle_left_square'])
lbl_middlemiddle=LabelEncoder()
dataset['middle_middle_square']=lbl_middlemiddle.fit_transform(dataset['middle_middle_square'])
lbl_middleright=LabelEncoder()
dataset['middle_right_square']=lbl_middleright.fit_transform(dataset['middle_right_square'])
lbl_bottomleft=LabelEncoder()
dataset['bottom_left_square']=lbl_bottomleft.fit_transform(dataset['bottom_left_square'])
lbl_bottommiddle=LabelEncoder()
dataset['bottom_middle_square']=lbl_bottommiddle.fit_transform(dataset['bottom_middle_square'])
lbl_bottomright=LabelEncoder()
dataset['bottom_right_square']=lbl_bottomright.fit_transform(dataset['bottom_right_square'])
lbl_class=LabelEncoder()
dataset['class']=lbl_class.fit_transform(dataset['class'])

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.svm import SVC
model=SVC()
model.fit(x_train,y_train)
y_predict=model.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_predict)
print(confusion)

print(model.score(x_test,y_test))