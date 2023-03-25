from asyncio.windows_events import NULL
from unicodedata import name
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('healthcare-dataset-stroke-data.csv')

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

# iterate through each column of df_catd and plot
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

# # Bieu do hinh tron
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


x=df.drop(['stroke'], axis=1)
y=df['stroke'] 
# print(df)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

models = dict()
models['KNN'] = KNeighborsClassifier()
models['Naive Bayes'] = GaussianNB()
models['Decision Tree'] = DecisionTreeClassifier()
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
list_knn=[]
list_bayes=[]
list_tree=[]


for index in range(10):
    # print(f"Number: {index+1}")
    for i in models:
        cnt=0
        total=0
        for a in range (0,10):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
            # print('-'*20+i+'-'*20)
            models[i].fit(x_train, y_train)
            model = models[i]
            y_pred = model.predict(x_test)
            arg_test = {'y_true':y_test, 'y_pred':y_pred}
            # print(confusion_matrix(**arg_test))
            # print(classification_report(**arg_test))
            total = total + accuracy_score(y_test, y_pred)*100
            cnt = cnt + 1
        avg = round(total / cnt, 2)
        if (i == 'KNN'):
            list_knn.append(avg)
        if (i == 'Naive Bayes'):
            list_bayes.append(avg)
        if (i == 'Decision Tree'):
            list_tree.append(avg)
        print(f"Average accuracy score {i}: ", avg)
        # print('-' * 40 )
    # print("\n")
plt.bar(range(10), list_knn, color ='#ff7f0e',width =0.2,edgecolor ='grey', label ='KNN')
plt.show()


# print(list_knn)
# print(list_bayes)
# print(list_tree)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)
# # print('-'*20+i+'-'*20)
# models['Naive Bayes'].fit(x_train, y_train)
# model = models['Naive Bayes']
# y_pred = model.predict(x_test)
# arg_test = {'y_true':y_test, 'y_pred':y_pred}
# print(confusion_matrix(**arg_test))
# print(classification_report(**arg_test))
