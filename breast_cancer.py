from sklearn.datasets import load_breast_cancer
dataset=load_breast_cancer()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(dataset.data,dataset.target,test_size=0.25,random_state=0)

from sklearn.svm import SVC
model=SVC()
model=SVC(kernel='linear')
model.fit(x_train,y_train)
y_predict=model.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_predict)

print(confusion)
print(model.score(x_test,y_test))