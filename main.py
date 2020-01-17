# Kullanılacak Kütüphanelerin İçe Aktarılması

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier

# Data import

df_test = pd.read_csv("C:/Users/Caner Filiz/Desktop/house-prices-advanced-regression-techniques/test.csv")
df_train = pd.read_csv("C:/Users/Caner Filiz/Desktop/house-prices-advanced-regression-techniques/train.csv")

# Summary of data
#pd.set_option('display.max_columns', None)
df_describe = df_train.describe().transpose()

# Detecting missing values
index = -1
list_features = []
list_missing_index = []
list_missing = []
for i in df_describe.transpose():
    list_features.append(i)
for i in df_describe['count']:
    index = index + 1
    if  i < 1460:
        list_missing_index.append(index)
        list_missing.append(i)
for i,j in zip(list_missing_index, list_missing):
    print("Değişken Türü: ",list_features[i],"Veri Sayısı: ", j)

print("*"*20,"Second Part","*"*20)


df_train = df_train[list_features]
list_features.remove('SalePrice')
df_test = df_test[list_features]
df_train.fillna(df_train.mean(), inplace=True)
df_test.fillna(df_test.mean(), inplace=True)


# Training data set
x_independent = df_train.drop("SalePrice", axis= 1)
y_dependent = df_train['SalePrice']
x_test = df_test

mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
mlp.fit(x_independent.values, y_dependent.values)
predicted_values = mlp.predict(x_test)
predicted_values = pd.DataFrame(predicted_values).transpose()
print(predicted_values.head(5))


f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(df_train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()

#corr_values = df_train.corr().abs()
#print(corr_values.iloc[-1].sort_values())





