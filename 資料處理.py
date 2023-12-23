import pandas as pd
df = pd.read_csv('clean_train.csv')
test = pd.read_csv('clean_test.csv')
test = test.iloc[:,1:]
train = df.iloc[:,1:]

y_train = train.Transported
x_train = train.drop(['Transported'], axis=1)
#共同處理分開的分析
ap_x=x_train.append(test)

#不使用填補資料，因使用的模型本身不在意這項問題
# One-Hot-Encoding
dummies = pd.get_dummies(ap_x.HomePlanet)
# Remove dummy trap
dummies_1 = dummies.iloc[:, 1:3]
ap_x = pd.concat([ap_x, dummies_1], axis=1)
ap_x = ap_x.drop(['HomePlanet'], axis=1)

# One-Hot-Encoding
dummies = pd.get_dummies(ap_x.Destination)
# Remove dummy trap
dummies_1 = dummies.iloc[:, 1:]
ap_x = pd.concat([ap_x, dummies_1], axis=1)
ap_x = ap_x.drop(['Destination'], axis=1)

# One-Hot-Encoding
dummies = pd.get_dummies(ap_x.abf)
# Remove dummy trap
dummies_1 = dummies.iloc[:, 1:8]
ap_x = pd.concat([ap_x, dummies_1], axis=1)
ap_x = ap_x.drop(['abf'], axis=1)


ap_x = ap_x.to_numpy()

from sklearn.preprocessing import LabelEncoder
import numpy as np
ap_x[:, 0] = LabelEncoder().fit_transform(ap_x[:, 0])
ap_x[:, 2] = LabelEncoder().fit_transform(ap_x[:, 2])
ap_x[:, 9] = LabelEncoder().fit_transform(ap_x[:, 9])
ap_x = np.asarray(ap_x).astype('float32')

#分開train_x與test
train_one = ap_x[:8693,:]
test_one = ap_x[8693:,:]

#normalize
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# Perform fit & transform to training data
train_one = sc.fit_transform(train_one)
# Perform transform to training data
test_one = sc.transform(test_one)
#test_one.to_csv('test_one.csv')
#train_one.to_csv('train_one.csv')
