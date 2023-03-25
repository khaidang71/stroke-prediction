import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
import warnings
warnings.filterwarnings(action='ignore')
data = pd.read_csv('healthcare-dataset-stroke-data.csv')
# print(data.head(3))

#Thay the gia tri null cua bmi thanh gia tri trung binh
data['bmi'] = data['bmi'].fillna( data['bmi'].mean() ) 
#Xoa cot ID
data = data.drop(columns ='id')
# print(data)
#Thay thuoc tinh other = thuoc tinh tieu bieu (xuat hien nhieu hon)
data['gender'] = data['gender'].replace('Other', list(data.gender.mode().values)[0])

df_cat = ['gender','hypertension','heart_disease','ever_married','work_type','Residence_type','smoking_status', 'stroke']

fig, axs = plt.subplots(4, 2, figsize=(14,20))
axs = axs.flatten()

#iterate through each column of df_catd and plot
# for i, col_name in enumerate(df_cat):
#     sns.countplot(x=col_name, data=data, ax=axs[i], hue =data['stroke'], palette = 'flare')
#     plt.title("Bar chart of")
#     axs[i].set_xlabel(f"{col_name}", weight = 'bold')
#     axs[i].set_ylabel('Count', weight='bold')
# plt.show()


# df_num = ['age', 'avg_glucose_level', 'bmi']
#
# fig, axs = plt.subplots(1, 3, figsize=(16,5))
# axs = axs.flatten()
#
# # iterate through each column in df_num and plot
# for i, col_name in enumerate(df_num):
#     sns.boxplot(x="stroke", y=col_name, data=data, ax=axs[i],  palette = 'Set1')
#     axs[i].set_xlabel("Stroke", weight = 'bold')
#     axs[i].set_ylabel(f"{col_name}", weight='bold')
# plt.show()

#Chi so BMI toi da la 50
data["bmi"] = pd.to_numeric(data["bmi"])
data["bmi"] = data["bmi"].apply(lambda x: 50 if x>50 else x)

#Bieu do hinh tron
# plt.figure(figsize=(4,4))
# data['stroke'].value_counts().plot.pie(autopct='%1.1f%%', colors = ['#66b3ff','#99ff99'])
# plt.title("Pie Chart of Stroke Status", fontdict={'fontsize': 14})
# plt.tight_layout()
# plt.show()

# Chuyen du lieu string thanh number
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data['gender'] = le.fit_transform(data['gender'])
data['ever_married'] = le.fit_transform(data['ever_married'])
data['work_type'] = le.fit_transform(data['work_type'])
data['Residence_type'] = le.fit_transform(data['Residence_type'])
data['smoking_status'] = le.fit_transform(data['smoking_status'])
df_en = data
# print(df_en.head())

#Xoa di ever_married vi tuong quan du lieu voi age
df_en = df_en.drop(['ever_married'], axis = 1)

#Chuan hoa du lieu: glucose, bmi, age
from sklearn.preprocessing import StandardScaler
s = StandardScaler()
columns = ['avg_glucose_level','bmi','age']
stand_scaled = s.fit_transform(df_en[['avg_glucose_level','bmi','age']])
stand_scaled = pd.DataFrame(stand_scaled,columns=columns)

df_en=df_en.drop(columns=columns,axis=1)
stand_scaled.head()
df = pd.concat([df_en, stand_scaled], axis=1)

#################################################
x=df.drop(['stroke'], axis=1)
y=df['stroke']

# Models
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Evaluation

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
# for i in range(1, 124):
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state= 124)

models = dict()
# models['kNN'] = KNeighborsClassifier()
models['Naive Bayes'] = GaussianNB()
for model in models:
    models[model].fit(x_train, y_train)
    print(model + " model fitting completed.")

print("Test Set Prediction:\n")

for x in models:
    print('-'*20+x+'-'*20)
    model = models[x]
    y_pred = model.predict(x_test)
    arg_test = {'y_true':y_test, 'y_pred':y_pred}
    print(confusion_matrix(**arg_test))
    print(classification_report(**arg_test))

# print('Summary of Accuracy Score\n\n')
# for i in models:
#     model = models[i]
#     print(i + ' Model: ',accuracy_score(y_test, model.predict(x_test)).round(4))


# Vong lap Kiem tra
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

kf = KFold(n_splits=11, shuffle= True, random_state= 1000)
total_acc_bayes = 0


for train_index, test_index in kf.split(x):
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("======================")
    # Bayes
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 4/10.0, random_state=5)
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy is ", accuracy_score(y_test, y_pred)*100)
    confusion_matrix(y_test, y_pred)
    total_acc_bayes += accuracy_score(y_test, y_pred)

print("Do chinh xac trung binh bayes: ",total_acc_bayes/60)