import pandas as pd
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
df = pd.read_csv('Churn_Modelling.csv')
df.head()
df.info()
df.describe()
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df['Geography'] = lb.fit_transform(df['Geography']) # type: ignore
df['Gender'] = lb.fit_transform(df['Gender']) # type: ignore
df.info()
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
df.shape
y = df.pop('Exited')
X = df
X.shape
y.shape
from sklearn.model_selection import train_test_split # type: ignore
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) # type: ignore
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train) # type: ignore
X_test = sc.fit_transform(X_test) # type: ignore
X_train # type: ignore
import keras # type: ignore
from keras.models import Sequential # type: ignore
from keras.layers import Dense # type: ignore
from keras.layers import LeakyReLU, PReLU, ELU # type: ignore
from keras.layers import Dropout # type: ignore
classifier = Sequential()
classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform',activation='relu',input_dim = 10)) # type: ignore
classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform',activation='relu')) # type: ignore
classifier.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid')) # type: ignore
classifier.compile(optimizer = 'Adamax', loss = 'binary_crossentropy', metrics = ['accuracy']) # type: ignore
model_history=classifier.fit(X_train, y_train, batch_size = 10, epochs = 100) # type: ignore
classifier.summary() # type: ignore
Y_pred = classifier.predict(X_test) # type: ignore
Y_pred = (Y_pred > 0.5) # type: ignore
print(Y_pred) # type: ignore
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, Y_pred) # type: ignore
print(cm)
from sklearn.metrics import accuracy_score
score=accuracy_score(Y_pred,y_test) # type: ignore
print(score)
#PLOTTING THE LOSS AND ACCURACY
plt.plot(model_history.history['loss']) # type: ignore
plt.title('Model Loss') # type: ignore
plt.ylabel('Loss') # type: ignore
plt.xlabel('Epoch') # type: ignore
plt.legend(['Train'], loc='upper right') # type: ignore
plt.show() # type: ignore
plt.plot(model_history.history['accuracy']) # type: ignore
plt.title('Model Accuracy') # type: ignore
plt.ylabel('Accuracy') # type: ignore
plt.xlabel('Epoch') # type: ignore
plt.legend(['Train'], loc='upper left') # type: ignore
plt.show() # type: ignore
#USE SEABORN TO PLOT THE LOSS AND ACCURACY
sns.set_style('whitegrid') # type: ignore
#PLOTTING THE LOSS AND ACCURACY
sns.lineplot(x=range(len(model_history.history['loss'])), y=model_history.history['loss']) # type: ignore
#Setting the title and labels
plt.title('Model Loss') # type: ignore
plt.ylabel('Loss') # type: ignore
plt.xlabel('Epoch') # type: ignore
plt.legend(['Train'], loc='upper right') # type: ignore
plt.show() # type: ignore