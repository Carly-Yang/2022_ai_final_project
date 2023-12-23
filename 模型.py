import pandas as pd
df = pd.read_csv('one_hot_train.csv')

test = pd.read_csv('one_hot_test.csv')
y_train = df.Transported
x_train = df.drop(['Transported'], axis=1)
test = test.iloc[:,1:]


# Importing the Keras libraries and packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow.keras.metrics as km
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import KFold

#KFold
kf = KFold(n_splits=10, shuffle=True, random_state = 42)
for tr_idx, va_idx in kf.split(x_train):
    tr_x, va_x = x_train.iloc[tr_idx], x_train.iloc[va_idx]
    tr_y, va_y = y_train[tr_idx], y_train[va_idx]
    print('訓練',len(tr_x))#7824
    print('val',len(va_x))#869
    
    
####### normal Classifier - 可不要跑###########
model = Sequential()
model.add(Dense(256,activation='relu',input_shape=(tr_x.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy', km.Precision(), km.Recall(), 'AUC'])
model.summary()
batch_size=128
epochs = 10
history = model.fit(tr_x, tr_y, batch_size=batch_size, epochs = epochs, verbose=1)
model.save("aimodel", save_format="tf")

#########直接預測即可#################
from tensorflow.keras.models import load_model
model = load_model("aimodel")
va_pred = model.predict(va_x)
tr_pred = model.predict(tr_x)
test_pred = model.predict(test)

y_pred_1 = (va_pred > 0.35)
tr_pred_1 = (tr_pred > 0.35)
test_pred_1 = (test_pred > 0.35)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
tr_ca = classification_report(tr_y, tr_pred_1)
cm = confusion_matrix(tr_y, tr_pred_1)#pred
acc = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])
acc

va_ca = classification_report(va_y, y_pred_1)
va_cm = confusion_matrix(va_y, y_pred_1)#pred
va_acc = (va_cm[0,0]+va_cm[1,1])/(va_cm[0,0]+va_cm[0,1]+va_cm[1,0]+va_cm[1,1])
va_acc

####### AutoKeras Classifier###########
import autokeras as ak
clf = ak.StructuredDataClassifier(max_trials=50, overwrite=True)
clf.fit(x=tr_x, y=tr_y, epochs=10)

clf_ca = classification_report(va_y, y_pred_1)
clf_cm = confusion_matrix(va_y, y_pred_1)#pred
clf_acc = (clf_cm[0,0]+clf_cm[1,1])/(clf_cm[0,0]+clf_cm[0,1]+clf_cm[1,0]+clf_cm[1,1])
clf_acc
# Save the best model
model.save("best_keras_aimodel", save_format="tf")

###########直接預測###############
# Reload the model to make predictions
loaded_model = load_model("best_keras_aimodel")

# convert NumPy array to Tensor for making predictions
prediction = loaded_model.predict(test)

from sklearn.metrics import confusion_matrix
predictions = (prediction > 0.35)
