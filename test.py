import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
import warnings
warnings.filterwarnings(action='ignore')

#a)
#Doc du lieu cua ruou vang do
data = pd.read_csv('healthcare-dataset-stroke-data.csv')
from sklearn.model_selection import train_test_split


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



X=df.drop(['stroke'], axis=1)
Y=df['stroke'] 
# print(df)


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


kf=KFold(n_splits=200, shuffle = True, random_state=500)
total = 0
cnt=0
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 4/10.0, random_state=50)
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    # print("==============================")
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 4/10.0, random_state=5)
    model = GaussianNB()
    model.fit(X_train, Y_train)
    Y_pred =  model.predict(X_test)
    total = total + accuracy_score(Y_test, Y_pred)*100
    print("Accuracy is ",cnt+1, accuracy_score(Y_test, Y_pred)*100)
    confusion_matrix(Y_test, Y_pred)
    cnt = cnt + 1
print("Do chinh xac trung binh: ", total/cnt)
