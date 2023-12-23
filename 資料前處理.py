import pandas as pd

dataset = pd.read_csv('raw_train.csv')
abf = dataset.Cabin.str[0]
num = dataset.Cabin.str[-3]
lr = dataset.Cabin.str[-1]
abf= abf.rename("abf")
num= num.rename("num")
lr= lr.rename("lr")

X = pd.concat([dataset, abf,num,lr], axis=1)

#dataset.to_csv('clean_train.csv')
